# quantum_nn.py
# 负责定义量子神经网络和相关组件

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def create_quantum_circuit():
    """
    创建量子电路：特征映射和参数化门
    
    返回:
    QuantumCircuit: 完整的量子电路
    feature_map: 特征映射部分
    ansatz: 参数化部分
    """
    # 创建特征映射电路
    feature_map = ZZFeatureMap(2)
    
    # 创建参数化电路
    ansatz = RealAmplitudes(2, reps=1)
    
    # 组合电路
    qc = QuantumCircuit(2)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    
    return qc, feature_map, ansatz

def create_qnn():
    """
    创建量子神经网络
    
    返回:
    EstimatorQNN: 量子神经网络对象
    """
    # 创建量子电路
    qc, feature_map, ansatz = create_quantum_circuit()
    
    # 创建估计器
    estimator = Estimator()
    
    # 创建可观测量 (Z ⊗ I)
    observable = SparsePauliOp.from_list([("ZI", 1.0)])
    
    # 创建量子神经网络
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
        observables=observable,
        input_gradients=True,
    )
    
    return qnn 