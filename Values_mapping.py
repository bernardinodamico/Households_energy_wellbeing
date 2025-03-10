'''
Values for each observed variable are represened as key-value pairs in a python dictionary.
the dict "key" corresponds to the "number" of the Variable's value (which is entered in the csv datased).
the dict "value" correspond to the "label" of the Variable's value

NOTE: real-valued Variables (e.g. Household income) are not mapped (empty dict). 
Only labelised variables (categorical, ordinal etc.) are mapped e.g. Dwelling age.
'''



# External walls insulation
X_values = {
    1: "Insulated wall (cavity/solid)",
    2: "Uninsulated wall (cavity/solid)"
}

# Heating energy (gas) use
Y_0_values = {}

# Mean rooms temperature
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
    9: "post 1990"
}

# Under Occupancy
V_4_values = {
    1: "Not under-occupied",
    2: "Under-occupied"
}