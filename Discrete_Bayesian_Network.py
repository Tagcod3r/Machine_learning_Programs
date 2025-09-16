from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([("Exercise", "HeartDisease"), ("Diet", "HeartDisease")])

cpd_exercise = TabularCPD("Exercise", 2, [[0.6], [0.4]])
cpd_diet = TabularCPD("Diet", 2, [[0.7], [0.3]])
cpd_hd = TabularCPD("HeartDisease", 2,
                    [[0.9, 0.6, 0.7, 0.1],
                     [0.1, 0.4, 0.3, 0.9]],
                    evidence=["Exercise", "Diet"],
                    evidence_card=[2, 2])

model.add_cpds(cpd_exercise, cpd_diet, cpd_hd)

infer = VariableElimination(model)
res = infer.query(variables=["HeartDisease"], evidence={"Exercise": 1, "Diet": 1})
print(res)
