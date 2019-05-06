from getFeatures import get_features
from getFeatures import concatenateData
from trainModel import trainPersonalizedMethod
from trainModel import testPersonalizedMethod


people = ["Raveh_Shulman", "Nadav_Finkel", "Raziel_Gartzman", "Oded_Kaplan", "Roy_Amir"]
#get_features(["Topaz_Enbar"], basic_override = True, euclid = True)
concatenateData(["Topaz_Enbar"], suffix = "_euc")

"""trainPersonalizedMethod(
    ["Oded_Kaplan", "Roy_Amir", "Topaz_Enbar", "Raziel_Gartzman", "Nadav_Finkel"],
    ["Raveh_Shulman"],
    "Logistic",
    suffix = "",
    label = "raveh_apart_reg",
    basic = True
)

testPersonalizedMethod("raveh_apart_reg", ["Raveh_Shulman"], suffix = '', basic = False)"""
