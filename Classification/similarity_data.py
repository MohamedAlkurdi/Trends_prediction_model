clusters = ['Economy','Technology and Science', 'Entertainment','Lifestyle','Accident','Geopolitical','Intellectualism']
countries = {'africa':['Kenya','Nigeria','SouthAfrica'],'europe':['Denmark','UK','Finland'],'north_america_australia':['Australia','Canada','USA'],'west_asia':['Malaysia','Philippines','Singapore']}

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
            "Denmark-Finland": 0,  
            "Denmark-UK": 0,  
            "Finland-UK": 0 
        },
        "Technology and Science": {
            "Denmark-Finland": 0,  
            "Denmark-UK": 0,  
            "Finland-UK": 0 
        },
        "Entertainment": {
            "Denmark-Finland": 0,  
            "Denmark-UK": 0,  
            "Finland-UK": 0 
        },
        "Lifestyle": {
            "Denmark-Finland": 0,  
            "Denmark-UK": 0,  
            "Finland-UK": 0 
        },
        "Accident": {
            "Denmark-Finland": 0,  
            "Denmark-UK": 0,  
            "Finland-UK": 0 
        },
        "Geopolitical": {
            "Denmark-Finland": 0,  
            "Denmark-UK": 0,  
            "Finland-UK": 0 
        },
        "Intellectualism": {
            "Denmark-Finland": 0,  
            "Denmark-UK": 0,  
            "Finland-UK": 0 
        },
    },
    "north_america_australia": {
        "Economy": {
            "Australia-Canada": 0,  
            "Australia-USA": 0,  
            "Canada-USA": 0 
        },
        "Technology and Science": {
            "Australia-Canada": 0,  
            "Australia-USA": 0,  
            "Canada-USA": 0 
        },
        "Entertainment": {
            "Australia-Canada": 0,  
            "Australia-USA": 0,  
            "Canada-USA": 0 
        },
        "Lifestyle": {
            "Australia-Canada": 0,  
            "Australia-USA": 0,  
            "Canada-USA": 0 
        },
        "Accident": {
            "Australia-Canada": 0,  
            "Australia-USA": 0,  
            "Canada-USA": 0 
        },
        "Geopolitical": {
            "Australia-Canada": 0,  
            "Australia-USA": 0,  
            "Canada-USA": 0 
        },
        "Intellectualism": {
            "Australia-Canada": 0,  
            "Australia-USA": 0,  
            "Canada-USA": 0 
        },
    },
    "west_asia": {
        "Economy": {
            "Malaysia-Philippines": 0,  
            "Malaysia-Singapore": 0,  
            "Philippines-Singapore": 0 
        },
        "Technology and Science": {
            "Malaysia-Philippines": 0,  
            "Malaysia-Singapore": 0,  
            "Philippines-Singapore": 0 
        },
        "Entertainment": {
            "Malaysia-Philippines": 0,  
            "Malaysia-Singapore": 0,  
            "Philippines-Singapore": 0 
        },
        "Lifestyle": {
            "Malaysia-Philippines": 0,  
            "Malaysia-Singapore": 0,  
            "Philippines-Singapore": 0 
        },
        "Accident": {
            "Malaysia-Philippines": 0,  
            "Malaysia-Singapore": 0,  
            "Philippines-Singapore": 0 
        },
        "Geopolitical": {
            "Malaysia-Philippines": 0,  
            "Malaysia-Singapore": 0,  
            "Philippines-Singapore": 0 
        },
        "Intellectualism": {
            "Malaysia-Philippines": 0,  
            "Malaysia-Singapore": 0,  
            "Philippines-Singapore": 0 
        },
    },
}