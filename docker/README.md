# Setup
To set up the environment, run `./setup.sh`.
This will build the docker environment and download all files.

# Using the docker environment
After the setup is complete, you should be able to enter the docker environment by running `docker/run.sh` (this assumes you are in the root of the project).

# Testing the setup
Run `docker/run.sh` and then inside the docker environment`python3 test.py`.
Alternatively you can run `docker/run.sh python3 test.py` to run the command in the docker environment.
The `test.py` script will show you the baseline performance.

# Generating adversarial voices
Run `docker/run.sh` and then `./attackMain.sh` or run `docker/run.sh ./attackMain.sh`.
You can find details in the [main reamde](../README.md).
