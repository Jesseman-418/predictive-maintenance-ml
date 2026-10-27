import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from pipeline import FEATURES, TARGET, ORDER, temporal_split, train, predict

def frame(n=100):
    rng=np.random.default_rng(4)
    df=pd.DataFrame({"UDI":np.arange(1,n+1),"Air temperature [K]":rng.normal(300,2,n),"Process temperature [K]":rng.normal(310,2,n),"Rotational speed [rpm]":rng.normal(1500,100,n),"Torque [Nm]":rng.normal(40,8,n),"Tool wear [min]":rng.integers(0,250,n),"Type":rng.choice(list("LMH"),n),"Machine failure":np.tile([0,0,0,1,0],n//5)})
    return df

class PipelineTests(unittest.TestCase):
    def test_temporal_split_is_ordered_and_disjoint(self):
        a,b,c=temporal_split(frame().sample(frac=1,random_state=7))
        self.assertLess(a.UDI.max(),b.UDI.min()); self.assertLess(b.UDI.max(),c.UDI.min())
    def test_train_and_predict(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            train(frame(),d)
            import joblib
            model=joblib.load(Path(d)/"model.joblib")
            result=predict(model,{k:frame().iloc[0][k] for k in FEATURES})
            self.assertGreaterEqual(result["failure_probability"],0)
            self.assertLessEqual(result["failure_probability"],1)
    def test_small_input_rejected(self):
        with self.assertRaises(ValueError): temporal_split(frame(10))

if __name__=="__main__": unittest.main()
