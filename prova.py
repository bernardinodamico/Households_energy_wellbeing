from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
import numpy as np
import pandas as pd

#data = pd.DataFrame(np.random.randint(low=0, high=2, size=(100, 3)), columns=['X', 'Y', 'Z'])

#data.to_csv(path_or_buf="DATA/prova_frequency_data.csv")

data = pd.read_csv(filepath_or_buffer="DATA/prova_frequency_data.csv")





model = BayesianNetwork([('Z', 'X'), ('Z', 'Y'), ('X', 'Y')])
model.fit(data)
inference = VariableElimination(model)

print("_____________________________")
print("P(Z)")
phi_query = inference.query(variables=['Z'], evidence=None)
print(phi_query)

print("_____________________________")
print("P(X | Z=0)")
phi_query = inference.query(variables=['X'], evidence={'Z': 0})
print(phi_query)

print("_____________________________")
print("P(X | Z=1)")
phi_query = inference.query(variables=['X'], evidence={'Z': 1})
print(phi_query)

#phi_query = inference.query(variables=['Y'], evidence=['X', 'Z'])
#print(phi_query)