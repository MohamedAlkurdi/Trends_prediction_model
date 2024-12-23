clusters = ['Economy','Technology and Science', 'Entertainment','Lifestyle','Accident','Geopolitical','Intellectualism']
countries = {'africa':['Kenya','Nigeria','SouthAfrica'],'europe':['Denmark','UK','Finland'],'north_america_australia':['Australia','Canada','USA'],'west_asia':['Malaysia','Philippines','Singapore']}
 
text ="claculated the relative traffic rate of the rest of the data files + started to calculate the similarity between the countries that belongs to the same region"
similarity_data = {
    "africa": {
        "Economy": {
            "Kenya-Nigeria": 99.370,  
            "Kenya-SouthAfrica": 98.10,  
            "Nigeria-SouthAfrica": 97.47 
        },
        "Technology and Science": {
            "Kenya-Nigeria": 80.38,  
            "Kenya-SouthAfrica": 81.01,  
            "Nigeria-SouthAfrica": 79.75, 
        },
        "Entertainment": {
            "Kenya-Nigeria": 60.13,  
            "Kenya-SouthAfrica": 60.76,  
            "Nigeria-SouthAfrica": 59.49 
        },
        "Lifestyle": {
            "Kenya-Nigeria": 70.89,  
            "Kenya-SouthAfrica": 58.23,  
            "Nigeria-SouthAfrica": 57.59 
        },
        "Accident": {
            "Kenya-Nigeria": 55.70,  
            "Kenya-SouthAfrica": 48.10,  
            "Nigeria-SouthAfrica": 49.37 
        },
        "Geopolitical": {
            "Kenya-Nigeria": 72.78,  
            "Kenya-SouthAfrica": 63.92,  
            "Nigeria-SouthAfrica": 62.03 
        },
        "Intellectualism": {
            "Kenya-Nigeria": 65.82,  
            "Kenya-SouthAfrica": 57.59,  
            "Nigeria-SouthAfrica": 57.59
        },
    },
    "europe": {
        "Economy": {
            "Denmark-Finland": 88.61,  
            "Denmark-UK": 92.41,  
            "Finland-UK": 93.67 
        },
        "Technology and Science": {
            "Denmark-Finland": 56.96,  
            "Denmark-UK": 53.80,  
            "Finland-UK": 50.63 
        },
        "Entertainment": {
            "Denmark-Finland": 86.71,  
            "Denmark-UK": 77.85,  
            "Finland-UK": 73.42 
        },
        "Lifestyle": {
            "Denmark-Finland": 48.10,  
            "Denmark-UK": 50.00,  
            "Finland-UK": 46.20 
        },
        "Accident": {
            "Denmark-Finland": 49.37,  
            "Denmark-UK": 51.90,  
            "Finland-UK": 60.76 
        },
        "Geopolitical": {
            "Denmark-Finland": 44.94,  
            "Denmark-UK": 48.73,  
            "Finland-UK": 43.67
        },
        "Intellectualism": {
            "Denmark-Finland": 44.30,  
            "Denmark-UK": 50.00,  
            "Finland-UK": 46.20
        },
    },
    "north_america_australia": {
        "Economy": {
            "Australia-Canada": 94.94,  
            "Australia-USA": 96.84,  
            "Canada-USA": 95.57 
        },
        "Technology and Science": {
            "Australia-Canada": 50.63,  
            "Australia-USA": 55.70,  
            "Canada-USA": 59.49 
        },
        "Entertainment": {
            "Australia-Canada": 91.14,  
            "Australia-USA": 85.44,  
            "Canada-USA": 91.77 
        },
        "Lifestyle": {
            "Australia-Canada": 68.35,  
            "Australia-USA": 72.15,  
            "Canada-USA": 65.19 
        },
        "Accident": {
            "Australia-Canada": 74.05,  
            "Australia-USA": 79.75,  
            "Canada-USA": 74.68 
        },
        "Geopolitical": {
            "Australia-Canada": 46.84,  
            "Australia-USA": 47.47,  
            "Canada-USA": 62.03
        },
        "Intellectualism": {
            "Australia-Canada": 58.23,  
            "Australia-USA": 70.25,  
            "Canada-USA": 55.70 
        },
    },
    "west_asia": {
        "Economy": {
            "Malaysia-Philippines": 95.57,  
            "Malaysia-Singapore": 93.67,  
            "Philippines-Singapore": 94.30
        },
        "Technology and Science": {
            "Malaysia-Philippines": 67.72,  
            "Malaysia-Singapore": 60.76,  
            "Philippines-Singapore": 63.29 
        },
        "Entertainment": {
            "Malaysia-Philippines": 78.48,  
            "Malaysia-Singapore": 81.65,  
            "Philippines-Singapore": 91.77 
        },
        "Lifestyle": {
            "Malaysia-Philippines": 62.66,  
            "Malaysia-Singapore": 52.53,  
            "Philippines-Singapore": 44.30 
        },
        "Accident": {
            "Malaysia-Philippines": 52.53,  
            "Malaysia-Singapore": 49.37,  
            "Philippines-Singapore": 47.47
        },
        "Geopolitical": {
            "Malaysia-Philippines": 76.58,  
            "Malaysia-Singapore": 55.06,  
            "Philippines-Singapore": 50.63 
        },
        "Intellectualism": {
            "Malaysia-Philippines": 58.86,  
            "Malaysia-Singapore": 66.46,  
            "Philippines-Singapore": 55.06 
        },
    },
}