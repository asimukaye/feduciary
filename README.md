# Feduciary: Fair Federated Learning in PyTorch

This repository is a modular Federated Learning sandbox in PyTorch for quick research prototyping. This package exposes abstract APIs required to model a FL strategy with the core design principle resembling around the Flower Framework with supporting modules for configuration management and results/logging. 

## Why not just use Flower?
While Flower allows for massive scaling and parallel processing capabilities, it comes at the cost of opacity and dependencies on the underlying libraries (flower is built on the Ray Library for parallel processing, for instance). Feduciary has a native sequential execution path for ease of debugging and complete whiteboxing with the possibility of using debuggers through the entire code pipeling. It is also helpful identifying unintentional side-effects of parallel execution, such as race conditions and or ordering errors. Further, the native code path also allows eases the passing of python objects without encountering serialization or pickling errors. Once the desired results are obtained , the existing strategy can be run on the Flower pipeline with a single mode switch to take advantage of the scalability that Flower provides. Layering its modules over flower also provides Feduciary the ability to run the packages on different devices.


The Feduciary package contains of the following modules:

- **Client**: The client module contains of submodules that define different types of clients available for training. The BaseClient Module is the simplest client module with a simple training and evaluation loop. The BaseFlowerClient is mixin of the BaseClient and The FlowerClient which enables the client to run within the Flower Framework. All the derived clients of BaseFlowerClient need not redefine the interfaces for Flower.

- **Strategy**: The strategy module is the highes level abstraction layer that describes the aggregation and selection strategy. It also contains the different I/O interfaces and configurations required fora a particular strategy. It is only required for a strategy developer to define the ins and outs of the strategy and provide a method of packaging it to be read by the client. The packing and unpacking strategies make the strategy module compatible with the underlying server which then takes care of the serialization and communication between itself and the clients.

- **Server**: The server module acts as the intermediary to communicate between the strategy and the clients. The BaseFlowerServer is a mixin of the feduciary server and the flower server.
- **Simulator**: The simulator module consisits of 4 modes of simulation typically required in Federated Learning Research:
    - *Centralized Mode*: Runs the entire dataset as a unified dataset with a single loop. This is required to establish a ceiling of the dataset and model in a non-federated setting.
    - *Federated Mode*: This is the native setting which runs the code in a federated setting with configurations to split the dataset in different patterns
    - *Standalone Mode*: The standalone mode provides a setting in which the federation of clients train on their respective datasets without a server aggregation step. This mode is required for fairness and incentive analysis for a particular strategy.
    - *Flower Mode*: The Flower Mode runs the federated learning algorithm with Flower simulator. Functionally, it is similar to the Federated Mode.

- **Results**: The results module contains the ResultManager that acts as a uniform logging interface for handling logging related functions, aggregations and interfacing with downstream points such as wandb, tensorboard and/or custom csv, json files.

- **Metrics**: The metrics module contains the MetricManager and metricszoo. MetricManager manages the different metrics that are required to be computed over the client training. It also provides a log to file capability for individual clients, which can be asynchronously processed by the ResultManager for emitting to downstread loggers.
- **Datasets**: The datasets module contains of the various dataset submodules that can be used for federated training
- **Models**: The models module contains model definition files that are used for federated training.
- **Common**: The common module contains of the typing and utils submodules that enable common utilities and custom types that are used throughout the package.

- **Config**: The config module serves as the entry point, schema validation and configuration validation for Feduciary. User defined configuration are defined as schemas and configs for validation against corresponding YAML files. The config module interfaces with config files using the HYDRA library. It also serves as a user input validation module and auto configurations script wherever necessary.

