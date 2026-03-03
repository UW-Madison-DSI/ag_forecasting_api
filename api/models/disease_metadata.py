from typing import Dict

DISEASE_MODELS: Dict[str, dict] = {
    "tarspot": {
        "name": "Tarspot",
        "description": (
            "Tar spot is caused by the fungus Phyllachora maydis, and can cause severe yield loss on susceptible hybrids when conditions are favorable for disease. Tar spot appears as small, raised, black spots scattered across the upper and lower leaf surfaces. These spots are stromata (fungal fruiting structures). If viewed under the microscope, hundreds of sausage-shaped asci (spore cases) filled with spores are visible. When severe, stromata can even appear on husks and leaf sheaths. Tan to brown lesions with dark borders surrounding stromata can also develop. These are known as “fisheye” lesions. In Latin America, where tar spot is more common, fisheye lesions are associated with another fungus, Monographella maydis, that forms a disease complex with P. maydis known as the tar spot complex. M. maydis has not been detected in the United States. At the end of the growing season, common and southern rust pustules can be mistaken for tar spot stromata as these rusts switch from producing orange-red spores (urediniospores) to black spores (teliospores). However, rust spores burst through the epidermis and the spores can be scraped away from the pustules with a fingernail while tar spots cannot be scraped off the leaf tissue. The pathogen that causes tar spot overwinters on infested corn residue on the soil surface, and it is thought that high relative humidity and prolonged leaf wetness favor disease development. Residue management, rotation, and avoiding susceptible hybrids may reduce tar spot development and severity. Some fungicides may also reduce tar spot, although little data exists regarding application timing for efficacy and economic response."
        ),
        "variables": [
            "air_temp_avg_c_30d_ma",
            "rh_max_30d_ma",
            "rh_above_90_night_14d_ma"
        ],
        "CPN": "https://cropprotectionnetwork.org/encyclopedia/tar-spot-of-corn",
        "model_type": "Logistic regression",
        "risk_output": "Probability scaled to 0–100",
        "inactive_rule": "Inactive when 30-day average temperature < threshold",
        "version": "1.0"
    },
    "frogeye_leaf_spot": {
        "name": "Frogeye Leaf Spot",
        "description": (
            "Frogeye leaf spot is caused by the fungus Cercospora sojina. The disease occurs across the United States and in Ontario, Canada. Frogeye leaf spot can cause significant yield loss when widespread within a field. Leaf lesions are small, irregular to circular in shape, and gray with reddish-brown borders. Most commonly occurring on the upper leaf surface, lesions start as dark, water-soaked spots that vary in size. As lesions age, the central area becomes gray to light brown with dark, red-brown margins. In severe cases, disease can cause premature leaf drop and will spread to stems and pods. The fungus survives in infested crop residue and infected seed. Early season infections contribute to infection of foliage and pods later in the season. Warm, humid weather promotes spore production, infection, and disease development. Young leaves are more susceptible to infection than older leaves, but visible lesions are not seen on young, expanding leaves because the lesions take two weeks to develop after infection. Resistant soybean varieties are available and should be used where disease is a potential problem. Several races of the pathogen have been identified, and varieties with resistance to all known races are available. Rotating to a non-host crop and tillage will reduce survival of C. sojina. Crops not susceptible to this pathogen are alfalfa, corn, and small grains. Foliar fungicides applied during late flowering and early pod set to pod filling stages can reduce frogeye leaf spot.If you believe fungicide resistance may be an issue in your field, contact your local extension specialist."
        ),
        "variables": [
            "air_temp_max_c_30d_ma",
            "rh_above_80_day_30d_ma"
        ],
        "CPN": "https://cropprotectionnetwork.org/encyclopedia/frogeye-leaf-spot-of-soybean",
        "model_type": "Logistic regression",
        "risk_output": "Probability scaled to 0–100",
        "version": "1.0"
    },
    "whitemold_irrigated": {
        "name": "White Mold (Irrigated)",
        "description": (
            "White mold (also called Sclerotinia stem rot) is a significant problem in the North Central soybean production region and Canada. Caused by the fungus Sclerotinia sclerotiorum, white mold is often recognized by fluffy, white growth on soybean stems. Initial symptoms generally develop from R3 to R6 as gray to white lesions at the nodes. Lesions rapidly progress above and below the nodes, sometimes girdling the stem. White, fluffy mold soon covers the infected area, especially during periods of high relative humidity. Characteristic black sclerotia eventually are visible and embedded within mold on stem lesions, and inside the stem as the plant approaches death. Initial foliar symptoms include leaf tissues between major veins turning a gray-green cast; eventually leaves die and turn completely brown while remaining attached to the stem. Pods affected by white mold generally contain seeds that are smaller, lighter, white, and cottony. Seeds may be replaced by sclerotia. It is unusual for an infected seed to look normal. Soybean seed lots can be contaminated with sclerotia, the survival structures produced by S. sclerotiorum in infected plants. The fungus survives in the soil for several years as sclerotia. The disease cycle begins when mushroom-like structures called apothecia are formed on the soil surface from sclerotia. Spores from apothecia infect senescing soybean flowers and the fungus eventually grows to the stem. The disease is more prevalent during cool, wet, or humid seasons and in fields where the canopy closes during soybean flowering and early pod development. No soybean variety is completely resistant to white mold, but varieties do range from moderately resistant to very susceptible. At least two or three years of a non-host crop can reduce the number of sclerotia in soil. Crops that should not be in rotation with soybean in fields with white mold risk are beans, peas, sunflowers, and cole crops (mustard or Brassicaceae family). More sclerotia are found near the soil surface in no-till systems, but sclerotia numbers begin to decline if left undisturbed. Early planting, narrow row width, and high plant populations all accelerate canopy closure and favor disease development. However, modification of these practices also may reduce yield potential. Weed control is critical as many broadleaf weeds are hosts of the white mold pathogen. Also, some herbicides may suppress the activity of the fungus or disrupt germination of sclerotia. Some antagonistic fungi may be applied to the soil to colonize and reduce sclerotia numbers. Foliar fungicides can manage white mold, or at least reduce disease severity; however, application timing is critical. Fungicides are most effective when applied immediately before infection."
        ),
        "variables": [
            "air_temp_max_c_30d_ma",
            "rh_max_30d_ma"
        ],
        "CPN": "https://cropprotectionnetwork.org/encyclopedia/white-mold-of-soybean",
        "model_type": "Logistic regression",
        "version": "1.0"
    },
    "whitemold_non_irrigated": {
        "name": "White Mold (Non-Irrigated)",
        "description": (
            "White mold (also called Sclerotinia stem rot) is a significant problem in the North Central soybean production region and Canada. Caused by the fungus Sclerotinia sclerotiorum, white mold is often recognized by fluffy, white growth on soybean stems. Initial symptoms generally develop from R3 to R6 as gray to white lesions at the nodes. Lesions rapidly progress above and below the nodes, sometimes girdling the stem. White, fluffy mold soon covers the infected area, especially during periods of high relative humidity. Characteristic black sclerotia eventually are visible and embedded within mold on stem lesions, and inside the stem as the plant approaches death. Initial foliar symptoms include leaf tissues between major veins turning a gray-green cast; eventually leaves die and turn completely brown while remaining attached to the stem. Pods affected by white mold generally contain seeds that are smaller, lighter, white, and cottony. Seeds may be replaced by sclerotia. It is unusual for an infected seed to look normal. Soybean seed lots can be contaminated with sclerotia, the survival structures produced by S. sclerotiorum in infected plants. The fungus survives in the soil for several years as sclerotia. The disease cycle begins when mushroom-like structures called apothecia are formed on the soil surface from sclerotia. Spores from apothecia infect senescing soybean flowers and the fungus eventually grows to the stem. The disease is more prevalent during cool, wet, or humid seasons and in fields where the canopy closes during soybean flowering and early pod development. No soybean variety is completely resistant to white mold, but varieties do range from moderately resistant to very susceptible. At least two or three years of a non-host crop can reduce the number of sclerotia in soil. Crops that should not be in rotation with soybean in fields with white mold risk are beans, peas, sunflowers, and cole crops (mustard or Brassicaceae family). More sclerotia are found near the soil surface in no-till systems, but sclerotia numbers begin to decline if left undisturbed. Early planting, narrow row width, and high plant populations all accelerate canopy closure and favor disease development. However, modification of these practices also may reduce yield potential. Weed control is critical as many broadleaf weeds are hosts of the white mold pathogen. Also, some herbicides may suppress the activity of the fungus or disrupt germination of sclerotia. Some antagonistic fungi may be applied to the soil to colonize and reduce sclerotia numbers. Foliar fungicides can manage white mold, or at least reduce disease severity; however, application timing is critical. Fungicides are most effective when applied immediately before infection."
        ),
        "variables": [
            "air_temp_max_c_30d_ma",
            "rh_max_30d_ma",
            "max_ws_30d_ma"
        ],
        "CPN": "https://cropprotectionnetwork.org/encyclopedia/white-mold-of-soybean",
        "model_type": "Logistic regression",
        "version": "1.0"
    },
    "gls_risk":{
        "name": "Gray Leaf spot",
        "description": (
            "Gray leaf spot, caused by the fungus Cercospora zeae-maydis, occurs virtually every growing season. If conditions favor disease development, economic losses can occur. Symptoms first appear on lower leaves about two to three weeks before tasseling. The leaf lesions are long (up to 2 inches), narrow, rectangular, and light tan colored. Later, the lesions can turn gray. They are usually delimited by leaf veins but can join together and kill entire leaves. The fungus survives in corn residue, and, consequently, the disease is often more severe in corn following corn. Spores are dispersed by wind and splashing water. Infection of corn leaves and disease development are favored by warm (80s°F), humid (>90% for 12+ hours) weather. Disease severity depends on hybrid susceptibility and environmental conditions. Resistant hybrids and inbreds are available. Crop rotation and tillage reduce survival of the fungus. Foliar fungicides labeled for gray leaf spot are available."
        ),
        "variables": [
        ],
        "CPN": "https://cropprotectionnetwork.org/encyclopedia/gray-leaf-spot-of-corn",
        "model_type": "Logistic regression",
        "version": "1.0"
    }
}