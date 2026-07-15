module D2DM_GUI_Constants
    export BAR1, BAR2, DESCR_LX, DESCR_LY, DESCR_LZ
    export ERUPTION_VOLUMES, ERUPTION_TIMES
    export DATA_FOLDER, D2DM_CAMPI_RHYOLITE_MF, D2DM_CAMPI_RHYOLITE_TEMP
    
    # For pretty logging
    const BAR1 = "\n├──"
    const BAR2 = "\n\t ├──"
    
    # Descriptions
    const DESCR_LX = "Lx\n\nThe size of the area along the x axis\n\nDimension: [m]"
    const DESCR_LY = "Ly\n\nThe size of the area along the y axis\n\nDimension: [m]"
    const DESCR_LZ = "Lz\n\nThe size of the area along the z axis\n\nDimension: [m]"
    
    # Default Volumes and times for Campi Flegrei
    const ERUPTION_VOLUMES = Float64[100, 154, 10, 10, 10, 10, 220, 45, 16, 50, 0.5, 0.02, 0.64, 0.02, 0.02, 0.7, 0.201, 0.06, 0.05, 0.02, 0.07, 0.930, 0.018, 0.12, 0.661, 0.016, 0.02, 0.029]
    const ERUPTION_TIMES = Float64[157.4, 109.3, 105.6, 102.5, 101.2, 91.8, 39.8, 39.7, 29.3, 14.9, 14.3, 13, 12, 12.8, 11.8, 11, 10.6, 9.6, 9.3, 5.1, 4.7, 4.9, 4.5, 4.3, 4.2, 4.1, 3.9, 0.5]
    
    # Data folder
    const DATA_FOLDER = "..\\d2dm_data\\"
    
    # Large arrays (const but contain immutable data)
    const D2DM_CAMPI_RHYOLITE_MF = Float64[1, 0.9978, 0.9955, 0.9918, 0.9881, 0.9401, 0.9273, 0.9231, 0.9189, 0.9029, 0.8859, 0.8532, 0.7972, 0.6777, 0.371, 0.2195, 0.1545, 0.1184, 0.1111, 0.1038, 0.0965, 0.0892, 0.0819, 0.0769, 0.0721, 0.0672, 0.0624, 0.0573, 0.0516, 0.0459, 0.0402, 0.0338, 0.0265, 0.0192, 0.0143, 0.0143, 0.0143, 0.0143, 0.0121, 0.0099, 0.0076, 0.006, 0.0056, 0.0051, 0.0047, 0.0043, 0.0037, 0.0028, 0.0019, 0.0009, 0]
    
    const D2DM_CAMPI_RHYOLITE_TEMP = Float64[699.2355, 708.9755, 718.7156, 728.4557, 738.1957, 747.9358, 757.6758, 767.4159, 777.156, 786.896, 796.6361, 806.3761, 816.1162, 825.8563, 835.5963, 845.3364, 855.0765, 864.8165, 874.5566, 884.2966, 894.0367, 903.7768, 913.5168, 923.2569, 932.9969, 942.737, 952.4771, 962.2171, 971.9572, 981.6972, 991.4373, 1001.1774, 1010.9174, 1020.6575, 1030.3976, 1040.1376, 1049.8777, 1059.6177, 1069.3578, 1079.0979, 1088.8379, 1098.578, 1108.318, 1118.0581, 1127.7982, 1137.5382, 1147.2783, 1157.0183, 1166.7584, 1176.4985, 1186.2385]
end