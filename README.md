# torchdata

**Under construction. Use at own risk.**

`torchdata` is a helper library for installing and reading from datasets.


## Configuration

Setting the `TORCHDATA_CACHE` environment variable causes torchdata to
retain downloaded files by storing them in the specified directory.


## Running tests

Some unit tests require access to a locally installed copy of the MPII Human Pose
dataset. To run these tests, set the `TORCHDATA_MPII_DATA_DIR` environment variable
to the appropriate location. If the environment variable is not set, dependent tests
will be skipped.

```bash
# Run all tests
$ TORCHDATA_MPII_DATA_DIR=/path/to/mpii/data pytest

# Run self-contained tests only
$ pytest
```


## License

(C) 2018 Aiden Nibali

This project is open source under the terms of the Apache License 2.0.

Please refer to the original websites for terms and conditions of using the datasets themselves.

* MPII Human Pose: [http://human-pose.mpi-inf.mpg.de/](http://human-pose.mpi-inf.mpg.de/)
