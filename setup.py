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
        #"neural-tangents @ git+ssh://git@github.com/rhaps0dy/neural-tangents.git@bb78dbf023febbfadaf1f4f791264ec7755e2996",
        "jaxlib @ https://storage.googleapis.com/jax-releases/cuda100/jaxlib-0.1.43-cp37-none-linux_x86_64.whl",
        "jax>=0.1.59<0.2",
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
