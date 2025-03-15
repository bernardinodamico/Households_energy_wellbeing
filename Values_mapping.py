
class VariableValues:

    @staticmethod
    def get_nums(var_symbol: str) -> list[str]:
        '''
        input: var_symbol = the symbol of the variable (e.g. X, Y_0 etc.)
        output: a list of the numerical values (e.g. 1, 2, 3...) for that variable
        '''
        var = Variables_dic[var_symbol]
        vals_list = []
        for v in list(var.keys()):
            vals_list.append(str(v))
        return vals_list
    
    @staticmethod
    def get_labels(var_symbol: str) -> list[str]:
        '''
        input: var_symbol = the symbol of the variable (e.g. X, Y_0 etc.)
        output: a list of the lable values (e.g. "End terrace", "Mid terrace"...) for that variable
        '''
        var = Variables_dic[var_symbol]
        vals_list = []
        for v in list(var.values()):
            vals_list.append(str(v))
        return vals_list
    

'''
Below are variables dictionaries: the values for each observed variable are represened as key-value pairs in a python dictionary.
the dict "key" corresponds to the "number" of the Variable's value (which is entered in the csv datased).
the dict "value" correspond to the "label" of the Variable's value.
'''

# External walls insulation
X_values = {
    1: "Insulated (cavity/solid) walls",
    2: "Uninsulated (cavity/solid) walls"
}

# Heating energy (gas) use
Y_0_values = {}

# Mean room temperature
Y_1_values = {}

# Heating energy burden
W_values = {
    1: 'Less than 2%',
    2: '2% to 3%',
    3: '3% to 4%',
    4: '4% to 5%',
    5: '5% to 6%',
    6: '6% to 7%',
    7: '7% to 8%',
    8: '8% to 9%',
    9: '9% to 10%',
    10: '10% to 11%',
    11: '11% to 12%',
    12: '12% to 13%',
    13: '13% to 14%',
    14: '14% to 15%',
    15: '15% to 16%',
    16: '16% and above',
}

# Fuel poverty
F_p_values = {
    0: "Not fuel poor",
    1: "Fuel poor"
}

# Dwelling type
V_0_values = {
    1: "End terrace",
    2: "Mid terrace",
    3: "Semi detached",
    4: "Detached",
    5: "Purpose built flat",
    6: "Converted flat"
}

# Space heating cost
V_1_values = {}

# Tenancy
V_2_values = {
    1: "Owner occupied",
    2: "Private rented",
    3: "Local authority",
    4: "Housing association"
}

# Dwelling age
V_3_values = {
    1: "Pre 1850",
    2: "1850 to 1899",
    3: "1900 to 1918",
    4: "1919 to 1944",
    5: "1945 to 1964",
    6: "1965 to 1974",
    7: "1975 to 1980",
    8: "1981 to 1990",
    9: "Post 1990"
}

# Under occupancy
V_4_values = {
    0: "Not under-occupied",
    1: "Under-occupied"
}

# Household size
V_5_values = {
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5 or more",
}

# Dwelling floor area
V_6_values = {
    1: "Less than 50 sqm",
    2: "50 to 69 sqm",
    3: "70 to 89 sqm",
    4: "90 to 109 sqm",
    5: "110 sqm or more",
}

# Household income
V_7_values = {
    1:	"Less than £15,000",
    2:	"£15,000 to £19,999",
    3:	"£20,000 to £29,999",
    4:	"£30,000 to £39,999",
    5:	"£40,000 to £49,999",
    6:	"£50,000 to £59,999",
    7:	"£60,000 to £69,999",
    8:	"£70,000 to £99,999",
    9:	"£100,000 and above"
}


# Household composition
V_8_values = {
    1: "Couple, no dependent child(ren) under 60",
    2: "Couple, no dependent child(ren) aged 60 or over",
    3: "Couple with dependent child(ren)",
    4: "Lone parent with dependent child(ren)",
    5: "Other multi-person households",
    6: "One person under 60",
    7: "One person aged 60 or over"
}




'''
Dictionary of all observed variables where: 
- dict-key = variable symbol
- dict-value = variable dictionary
'''
Variables_dic = {
    'X': X_values,
    'Y_0': Y_0_values,
    'Y_1': Y_1_values,
    'W': W_values,
    'V_0': V_0_values,
    'V_1': V_1_values,
    'V_2': V_2_values,
    'V_3': V_3_values,
    'V_4': V_4_values,
    'V_5': V_5_values,
    'V_6': V_6_values,
    'V_7': V_7_values,
    'V_8': V_8_values,
}