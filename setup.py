from setuptools import setup

setup(
    name="cnn_limits",
    version="0.1.0",
    description="Correlated limits of CNNs",
    long_description="",
    author="Adrià Garriga-Alonso",
    author_email="adria.garriga@gmail.com",
    url="https://github.com/rhaps0dy/cnn-limits/",
    license="Apache License 2.0",
    packages=["cnn_limits"],
    install_requires=[
        #"cnn-gp @ git+ssh://git@github.com/cambridge-mlg/cnn-gp.git@b61bc01f8e2dfd2427798793a3677c5b1a6a706c",
        #"neural-tangents @ git+ssh://git@github.com/rhaps0dy/neural-tangents.git@bb78dbf023febbfadaf1f4f791264ec7755e2996",
        "jaxlib @ https://storage.googleapis.com/jax-releases/cuda100/jaxlib-0.1.43-cp37-none-linux_x86_64.whl",
        "jax>=0.1.59<0.2",
        #"nigp @ git+ssh://git@github.com/cambridge-mlg/nigp.git@6ac4f3fc8789d9fb6c9f94115810a445b61e3aeb",
        "gpytorch>=1.0.1<1.1",
        "numpy>=1.18<1.19",
        "pickle-utils==0.1",
        "sacred>=0.8<0.9",
        "tensorboardX>=2.0",
        "torch>=1.4<1.5",
        "torchvision>=0.5<0.6",
        "h5py>=2.10<2.11",
    ],
    test_suite="testing",
)
