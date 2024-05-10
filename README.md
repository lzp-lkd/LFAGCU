# LFAGCU
Manuscript (Local Feature Acquisition and Global Context Understanding Network for Very High-Resolution Land Cover Classification) code description

** Model.py ** : is the model file
**train.py** : is the file that calls the model training
**predict.py** : is a file that calls the model to make predictions
** class_RSSCN7.json**,**class_UCMerced_LandUse.json**,**class_WHU_RS19.json** : is the label file corresponding to the training data set 
 """    RSSCN7 data    """
    legend_labels = ["aGrass",
                  "bField",
                  "cIndustry",
                  "dRiverLake",
                  "eForest",
                  "fResident",
                  "gParking"]

    """    WHU_RS19 data    """
    # legend_labels = [
    #     "Airport", "Beach", "Bridge", "Commercial", "Desert", "Farmland", "Football Field",
    #     "Forest", "Industrial", "Meadow", "Mountain", "Park", "Parking", "Pond", "Port",
    #     "Railway Station", "Residential", "River", "Viaduct"
    # ]
    """
    UCMerced_LandUse data
    """
    legend_labels = [
        "agricultural",
        "airplane",
        "baseballdiamond",
        "beach",
        "buildings",
        "chaparral",
        "denseresidential",
        "forest",
        "freeway",
        "golfcourse",
        "harbor",
        "intersection",
        "mediumresidential",
        "mobilehomepark",
        "overpass",
        "parkinglot",
        "river",
        "runway",
        "sparseresidential",
        "storagetanks",
        "tenniscourt",
    ]

