from .flow_vbayes import FlowVBayes
from .planar_flow import PlanarFlow
from .target_toy import (
    TargetToy1, 
    TargetToy2, 
    plot_contour_and_surface3d,
)
from .target_glm import (
    TargetBayesianRegression,
    LinearRegressionLikelihood,
    LogisticRegressionLikelihood,
    UniformCauchyPrior,
    DGP,
    estimate_log_evidence,
)

