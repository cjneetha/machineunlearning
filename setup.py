import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='machineunlearning',  
     version='1.0',
     scripts=[''] ,
     author="Neetha Jambigi",
     author_email="cjneetha@gmail.com",
     description="A package for machine unlearning",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )