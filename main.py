from getFeatures import get_features
from getFeatures import concatenateData
from trainModel import personalizedMethod

people = ["Roy_Amir", "Topaz_Enbar", "Raveh_Shulman", "Nadav_Finkel", "Raziel_Gartzman"]#, "Oded_Kaplan"]
get_features(people)
#concatenateData(people)

personalizedMethod(
    ["Oded_Kaplan", "Roy_Amir", "Topaz_Enbar", "Raveh_Shulman", "Nadav_Finkel"],
    ["Raziel_Gartzman"],
    "Linear"
)
