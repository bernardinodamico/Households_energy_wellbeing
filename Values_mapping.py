'''
Values for each observed variable are represened as key-value pairs in a python dictionary.
the dict "key" corresponds to the "number" of the Variable's value (which is entered in the csv datased).
the dict "value" correspond to the "label" of the Variable's value

NOTE: real-valued Variables (e.g. Household income) are not mapped (empty dict). 
Only labelised variables (categorical, ordinal etc.) are mapped e.g. Dwelling age.
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
W_values = {}

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
    1: "Not under-occupied",
    2: "Under-occupied"
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
V_7_values = {}

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
