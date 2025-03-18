
def set_discrete_range_and_bounds(lower_bond, upper_bond, bins_num, round_by):
    
    bins_intervals = []
    bins_intervals.append(0.)
    incr = (upper_bond - lower_bond) / (bins_num - 2)
    for i in range(0, bins_num-1):
        bins_intervals.append(round(lower_bond + incr * i, round_by))  
    bins_intervals.append(1000000.)


    Var_values = {}
    for i in range(0, bins_num):
        if i == 0:
            Var_values[int(i+1)] = f"< {bins_intervals[i+1]}"
        elif i == bins_num - 1:
            Var_values[int(i+1)] = f"> {bins_intervals[i]}"
        else:
            Var_values[int(i+1)] = f"{bins_intervals[i]} to {bins_intervals[i+1]}"

    return [bins_intervals, Var_values]


class GetVariableValues:

    @staticmethod
    def get_nums(var_symbol: str) -> list[str]:
        '''
        input: var_symbol = the symbol of the variable (e.g. X, Y_0 etc.)
        output: a list of the numerical values (e.g. 1, 2, 3...) for that variable
        '''
        vv_dic = VariableValues()
        var = vv_dic.Variables_dic[var_symbol]
        vals_list = []
        for v in list(var.keys()):
            vals_list.append(str(v))
        return vals_list
    

    @staticmethod
    def get_labels(var_symbol: str) -> list[str]:
        '''
        input: var_symbol = the symbol of the variable (e.g. X, Y_0 etc.)
        output: a list of the label values (e.g. "End terrace", "Mid terrace"...) for that variable
        '''
        vv_dic = VariableValues()
        var = vv_dic.Variables_dic[var_symbol]
        vals_list = []
        for v in list(var.values()):
            vals_list.append(str(v))
        return vals_list
    

    @staticmethod
    def get_bins_intervals(var_symbol: str) -> list[str]:
        '''
        NOTE: Only for real-valued variables
        input: var_symbol = the symbol of the variable (e.g. W, Y_0 etc.)
        output: a list of pandas Intervals (e.g. (3.5, 4], (4, 4.5]...)) for that variable
        '''
        vv_dic = VariableValues()
        return vv_dic.Variables_bins_intervals[var_symbol]
 

class VariableValues():
    '''
    Variables dictionaries: the values for each observed variable are represened as key-value pairs in a python dictionary.
    - The dict-key corresponds to the "number" of the Variable's value (which is entered in the csv datased).
    - The dict-value correspond to the "label" of the Variable's value. 


    Discrete variables (categorical, ordinal, etc.)
    '''
    # External walls insulation
    X_values = {
        1: "Insulated (cavity/solid) walls",
        2: "Uninsulated (cavity/solid) walls"
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

    # Tenancy
    V_2_values = {
        1: "Owner occupied",
        2: "Private rented",
        3: "Local authority",
        4: "Housing association"
    }

    # Dwelling age
    V_3_values = {
        1: "< 1850",
        2: "1850 to 1899",
        3: "1900 to 1918",
        4: "1919 to 1944",
        5: "1945 to 1964",
        6: "1965 to 1974",
        7: "1975 to 1980",
        8: "1981 to 1990",
        9: "> 1990"
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
        1: "< 50 sqm",
        2: "50 to 70 sqm",
        3: "70 to 90 sqm",
        4: "90 to 110 sqm",
        5: "> 110 sqm",
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
    Real-valued variables (to discretise)
    '''

    discrete_setting_Y_0 = set_discrete_range_and_bounds(lower_bond=11700., upper_bond=17000., bins_num=6, round_by=0)
    discrete_setting_Y_1 = set_discrete_range_and_bounds(lower_bond=18.4, upper_bond=19.4, bins_num=8, round_by=3)
    discrete_setting_W = set_discrete_range_and_bounds(lower_bond=0.02, upper_bond=0.08, bins_num=5, round_by=4)
    discrete_setting_V_1 = set_discrete_range_and_bounds(lower_bond=430, upper_bond=750, bins_num=8, round_by=0)
    discrete_setting_V_7 = set_discrete_range_and_bounds(lower_bond=10000, upper_bond=99999.999, bins_num=11, round_by=0)

    # Energy (gas) consumption
    Y_0_values = discrete_setting_Y_0[1]

    # Indoor temperature
    Y_1_values = discrete_setting_Y_1[1]

    # Energy burden
    W_values = discrete_setting_W[1]

    # Household income
    V_7_values = discrete_setting_V_7[1]

    # Gas cost
    V_1_values = discrete_setting_V_1[1]


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

    Variables_bins_intervals = {
        'Y_0': discrete_setting_Y_0[0],
        'Y_1': discrete_setting_Y_1[0],
        'W': discrete_setting_W[0],
        'V_1': discrete_setting_V_1[0],
        'V_7': discrete_setting_V_7[0],
    }

