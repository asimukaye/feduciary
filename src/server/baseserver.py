from abc import ABCMeta, abstractmethod
import os
import logging
import json
import torch
import random
import gc
import numpy as np

from src import init_weights, TqdmToLogger, MetricManager
from collections import ChainMap, defaultdict
from importlib import import_module
from concurrent.futures import ThreadPoolExecutor, as_completed


# TODO: Long term todo: the server should probably eventually be tied directly to server algorithm
logger = logging.getLogger(__name__)

class BaseServer(metaclass=ABCMeta):
    """Centeral server orchestrating the whole process of federated learning.
    """
    def __init__(self, args, writer, server_dataset, client_datasets, model,  **kwargs):
        self.round = 0
        self._model = model
        self._clients = None
        self.args = args
        self.writer = writer

        # global holdout set
        if self.args.eval_type != 'local':
            self.server_dataset = server_dataset

        # clients
        # Why should the server create the clients?
        # TODO: Probably need to disambiguate server from simulator
        self._clients = self._create_clients(client_datasets)
        
        self.results = defaultdict(dict)


    def add_client(self, id, client):
        # TODO: function to add a client to the map of clients
        pass

    def remove_client(self, id):
        # TODO: remove a client by id
        pass

    def _init_model(self, model):
        # initialize the model class 
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Initialize a model!')
        init_weights(model, self.args.init_type, self.args.init_gain)
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully initialized the model ({self.args.model_name}; (Initialization type: {self.args.init_type.upper()}))!')
        return model

    def _get_algorithm(self, model, **kwargs):
        # This looks like a futile function as it is highly tied to the server type. It's best imported. Will retain only for legacy purposes.
        # Imports the algorithm from algorithms modules
        ALGORITHM_CLASS = import_module(f'..algorithm.{self.args.algorithm}', package=__package__).__dict__[f'{self.args.algorithm.title()}Optimizer']
        # 
        return ALGORITHM_CLASS(params=model.parameters(), **kwargs)


    def _create_clients(self, client_datasets):
        # NOTE: This can be moved out of the server function eventually
        # Acess the client class
        CLIENT_CLASS = import_module(f'..client.{self.args.algorithm}client', package=__package__).__dict__[f'{self.args.algorithm.title()}Client']

        def __create_client(identifier, datasets):
            client = CLIENT_CLASS(args=self.args, training_set=datasets[0], test_set=datasets[-1])
            client.id = identifier
            return client

        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Create clients!')
        
        # FIXME: separate submit from results call or remove concurrency all together
        clients = {}
        with ThreadPoolExecutor(max_workers=min(int(self.args.K), os.cpu_count() - 1)) as workhorse:
            for identifier, datasets in TqdmToLogger(
                enumerate(client_datasets), logger=logger, 
                desc=f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...creating clients... ',
                total=len(client_datasets)):
                clients[identifier] = workhorse.submit(__create_client, identifier, datasets).result()

        # print(clients[0].download)
        
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully created {self.args.K} clients!')
        return clients

    # @abstractmethod
    def _broadcast_models(self, ids):
        """broadcast the global model to all the clients.
        TODO: finish the documentation of this function
        Args:
            ids (_type_): client ids
        """
        def __broadcast_model(client):
            client.download(self.model)
        
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Broadcast the global model at the server!')
        
        # FIXME: if this call is fast, no need for concurrency
        self.model.to('cpu')
        with ThreadPoolExecutor(max_workers=min(len(ids), os.cpu_count() - 1)) as workhorse:
            for identifier in TqdmToLogger(
                ids, 
                logger=logger, 
                desc=f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...broadcasting server model... ',
                total=len(ids)):
                workhorse.submit(__broadcast_model, self._clients[identifier]).result()
      
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...sucessfully broadcasted the model to selected {len(ids)} clients!')

    # @abstractmethod
    def _sample_clients(self, exclude=[]):
        # NOTE: Update does not use the logic of C+ 0 meaning all clients
        
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Sample clients!')
        
        # Update - randomly select max(floor(C * K), 1) clients
        if exclude == []: 
            num_sampled_clients = max(int(self.args.C * self.args.K), 1)
            sampled_client_ids = sorted(random.sample([i for i in range(self.args.K)], num_sampled_clients))

        # Evaluation - randomly select unparticipated clients in amount of `eval_fraction` multiplied
        else: 
            num_unparticipated_clients = self.args.K - len(exclude)
            if num_unparticipated_clients == 0: # when C = 1, i.e., need to evaluate on all clients
                num_sampled_clients = self.args.K
                sampled_client_ids = sorted([i for i in range(self.args.K)])
            else:
                num_sampled_clients = max(int(self.args.eval_fraction * num_unparticipated_clients), 1)
                sampled_client_ids = sorted(random.sample([identifier for identifier in [i for i in range(self.args.K)] if identifier not in exclude], num_sampled_clients))
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...{num_sampled_clients} clients are selected!')
        return sampled_client_ids

    
    def _request(self, ids, eval=False, participated=False):
        # TODO: maybe this can be split into two functions
        def __update_clients(client):
            # getter function for client update
            client.args.lr = self.lr_scheduler.get_last_lr()[-1]
            update_result = client.update()
            return {client.id: len(client.training_set)}, {client.id: update_result}

        def __evaluate_clients(client):
            # getter function for client evaluate
            eval_result = client.evaluate() 
            return {client.id: len(client.test_set)}, {client.id: eval_result}

        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Request {"updates" if not eval else "evaluation"} to {"all" if ids is None else len(ids)} clients!')
        if eval:
            if self.args._train_only: return
            results = []
            with ThreadPoolExecutor(max_workers=min(len(ids), os.cpu_count() - 1)) as workhorse:
                for idx in TqdmToLogger(
                    ids, 
                    logger=logger, 
                    desc=f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...evaluate clients... ',
                    total=len(ids)
                    ):
                    # FIXME: separate results from submit
                    results.append(workhorse.submit(__evaluate_clients, self._clients[idx]).result()) 
            eval_sizes, eval_results = list(map(list, zip(*results)))
            eval_sizes, eval_results = dict(ChainMap(*eval_sizes)), dict(ChainMap(*eval_results))
            self.results[self._round][f'clients_evaluated_{"in" if participated else "out"}'] = self._log_results(
                eval_sizes, 
                eval_results, 
                eval=True, 
                participated=participated
            )
            logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...completed evaluation of {"all" if ids is None else len(ids)} clients!')
        else:
            results = []
            futures = []

            # Test dithing concurrency
            for idx in TqdmToLogger(
                    ids, 
                    logger=logger, 
                    desc=f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...receiving updates... ',
                    total=len(ids)
                    ):
                    # Client accessed here
                    # print("Got result for client: ", furtures[future])
                    results.append(__update_clients(self._clients[idx]))

            # with ThreadPoolExecutor(max_workers=min(len(ids), os.cpu_count() - 1)) as workhorse:
            #     futures = [workhorse.submit(__update_clients, self._clients[idx]) for idx in ids]
                # for idx in TqdmToLogger(
                #     ids, 
                #     logger=logger, 
                #     desc=f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...update clients... ',
                #     total=len(ids)
                #     ):
                #     # Client accessed here
                #     futures.append(workhorse.submit(__update_clients, self._clients[idx])) 

                # for future in TqdmToLogger(
                #     as_completed(futures), 
                #     logger=logger, 
                #     desc=f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...receiving updates... ',
                #     total=len(ids)
                #     ):
                #     # Client accessed here
                #     # print("Got result for client: ", furtures[future])
                #     results.append(future.result())

                # for i, future in enumerate(futures):
                #     print("Got result for client: ", i)
                #     results.append(future.result())


            # TODO: See what is happening here? 
            # print(type(results))
            update_sizes, update_results = list(map(list, zip(*results)))

            update_sizes, update_results = dict(ChainMap(*update_sizes)), dict(ChainMap(*update_results))

            # print('\n------------------------\n')
   
            self.results[self._round]['clients_updated'] = self._log_results(
                update_sizes, 
                update_results, 
                eval=False, 
                participated=True
            )
            logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...completed updates of {"all" if ids is None else len(ids)} clients!')
            return update_sizes
    

    def _cleanup(self, indices):
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Clean up!')

        for identifier in indices:
            if self._clients[identifier].model is not None:
                self._clients[identifier].model = None
            else:
                err = f'why clients ({identifier}) has no model? please check!'
                logger.exception(err)
                raise AssertionError(err)
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...successfully cleaned up!')
        gc.collect()

    @torch.inference_mode()
    def _central_evaluate(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.eval()
        self.model.to(self.args.device)

        for inputs, targets in torch.utils.data.DataLoader(dataset=self.server_dataset, batch_size=self.args.B, shuffle=False):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            outputs = self.model(inputs)
            loss = torch.nn.__dict__[self.args.criterion]()(outputs, targets)

            mm.track(loss.item(), outputs, targets)
        else:
            mm.aggregate(len(self.server_dataset))

        # log result
        result = mm.results
        server_log_string = f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] [EVALUATE] [SERVER] '

        ## loss
        loss = result['loss']
        server_log_string += f'| loss: {loss:.4f} '
        
        ## metrics
        for metric, value in result['metrics'].items():
            server_log_string += f'| {metric}: {value:.4f} '
        logger.info(server_log_string)

        # log TensorBoard
        self.writer.add_scalar('Server Loss', loss, self.round)
        for name, value in result['metrics'].items():
            self.writer.add_scalar(f'Server {name.title()}', value, self.round)
        else:
            self.writer.flush()
        self.results[self.round]['server_evaluated'] = result


    def finalize(self):
        """Save results.
        """
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Save results and the global model checkpoint!')
        # save figure
        with open(os.path.join(self.args.result_path, f'{self.args.exp_name}.json'), 'w', encoding='utf8') as result_file:
            results = {key: value for key, value in self.results.items()}
            json.dump(results, result_file, indent=4)

        # save checkpoint
        torch.save(self.model.state_dict(), os.path.join(self.args.result_path, f'{self.args.exp_name}.pt'))
        
        self.writer.close()

        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...finished federated learning!')


    def _log_results(self, resulting_sizes, results, eval, participated):
        losses, metrics, num_samples = list(), defaultdict(list), list()
        for identifier, result in results.items():
            client_log_string = f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [CLIENT] < {str(identifier).zfill(6)} > '

            # get loss and metrics
            if eval:
                # loss
                loss = result['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)
                
                # metrics
                for metric, value in result['metrics'].items():
                    client_log_string += f'| {metric}: {value:.4f} '
                    metrics[metric].append(value)
            else: # same, but retireve results of last epoch's
                # loss
                loss = result[self.args.E]['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)
                
                # metrics
                for name, value in result[self.args.E]['metrics'].items():
                    client_log_string += f'| {name}: {value:.4f} '
                    metrics[name].append(value)                
            # get sample size
            num_samples.append(resulting_sizes[identifier])

            # log per client
            logger.info(client_log_string)
        else:
            num_samples = np.array(num_samples).astype(float)

        # aggregate intototal logs
        result_dict = defaultdict(dict)
        total_log_string = f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [SUMMARY] ({len(resulting_sizes)} clients):'

        # loss
        losses = np.array(losses).astype(float)
        weighted = losses.dot(num_samples) / sum(num_samples)
        equal = losses.mean()
        std = losses.std()
        total_log_string += f'\n    - Loss: Weighted Avg. ({weighted:.4f}) | Equal Avg. ({equal:.4f}) | Std. ({std:.4f}) |'
        result_dict['loss'] = {'weighted': weighted, 'equal': equal, 'std': std}
        self.writer.add_scalars(
            f'Local {"Test" if eval else "Training"} Loss ' + eval * f'({"In" if participated else "Out"})',
            {f'Weighted Average': weighted, f'Equal Average': equal},
            self.round
        )

        # metrics
        for name, val in metrics.items():
            val = np.array(val).astype(float)
            weighted = val.dot(num_samples) / sum(num_samples)
            equal = val.mean()
            std = val.std()
            total_log_string += f'\n    - {name.title()}: Weighted Avg. ({weighted:.4f}) | Equal Avg. ({equal:.4f}) | Std. ({std:.4f}) |'
            result_dict[name] = {'weighted': weighted, 'equal': equal, 'std': std}
            for name in metrics.keys():
                self.writer.add_scalars(
                    f'Local {"Test" if eval else "Training"} {name.title()}' + eval * f'({"In" if participated else "Out"})',
                    {f'Weighted Average': weighted, f'Equal Average': equal},
                    self.round
                )
            self.writer.flush()
        
        # log total message
        logger.info(total_log_string)
        return result_dict


    def evaluate(self, excluded_ids):
        """Evaluate the global model located at the server.
        """
        # randomly select all remaining clients not participated in current round
        selected_ids = self._sample_clients(exclude=excluded_ids)
        self._broadcast_models(selected_ids)

        # request evaluation 
        ## `local`: evaluate on selected clients' holdout set
        ## `global`: evaluate on the server's global holdout set 
        ## `both`: conduct both `local` and `global` evaluations
        if self.args.eval_type == 'local':
            self._request(selected_ids, eval=True, participated=False)
        elif self.args.eval_type == 'global':
            self._central_evaluate()
        elif self.args.eval_type == 'both':
            self._request(selected_ids, eval=True, participated=False)
            self._central_evaluate()

        # remove model copy in clients
        self._cleanup(selected_ids)

        # calculate generalization gap
        if (not self.args._train_only) and (not self.args.eval_type == 'global'):
            gen_gap = dict()
            curr_res = self.results[self.round]
            for key in curr_res['clients_evaluated_out'].keys():
                for name in curr_res['clients_evaluated_out'][key].keys():
                    if name in ['equal', 'weighted']:
                        gap = curr_res['clients_evaluated_out'][key][name] - curr_res['clients_evaluated_in'][key][name]
                        gen_gap[f'gen_gap_{key}'] = {name: gap}
                        self.writer.add_scalars(f'Generalization Gap ({key.title()})', gen_gap[f'gen_gap_{key}'], self.round)
                        self.writer.flush()
            else:
                self.results[self.round]['generalization_gap'] = dict(gen_gap)


    # Evert server needs to implement these uniquely

    @abstractmethod
    def _aggregate(self, indices, update_sizes):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError
