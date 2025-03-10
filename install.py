import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

# Define the R packages to be installed
packages_to_install = StrVector(['grf', 'policytree'])

# Install the packages
utils = rpackages.importr('utils')
utils.install_packages(packages_to_install)