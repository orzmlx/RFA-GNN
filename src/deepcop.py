import tensorflow as tf
from tensorflow.keras import layers

class DeepCOP(tf.keras.Model):
    """
    DeepCOP Baseline Model Implementation
    Reference: DeepCOP (Deep Learning for Cell-specific Omics Prediction)
    
    Architecture:
    - Input: Concatenation of Control Expression (Gene Features) and Drug Fingerprint
    - Hidden Layers: MLP with BatchNormalization and Dropout
    - Output: Predicted Treated Expression (Regression)
    
    Key Features from Original Paper/Code:
    - Hidden Dimension = Input Dimension (Wide layers)
    - Activation: SELU (Layer 1) -> ReLU (Layer 2)
    - Regularization: BN + Dropout (0.2)
    """
    
    def __init__(
        self,
        num_genes,
        drug_dim=2048,
        dropout=0.2,
        use_residual=True,
        go_matrix=None,
        original_architecture=False,
        hidden_dim=1024,
        predict_uncertainty=False,
        **kwargs,
    ):
        super(DeepCOP, self).__init__(**kwargs)
        self.use_residual = use_residual
        self.num_genes = num_genes
        self.drug_dim = int(drug_dim)
        self.dropout_rate = float(dropout)
        self.original_architecture = bool(original_architecture)
        self.predict_uncertainty = bool(predict_uncertainty)
        
        # 处理 GO Matrix
        self.go_matrix = None
        go_dim = 0
        if go_matrix is not None:
            # 转换为常量 Tensor，无需训练
            self.go_matrix = tf.constant(go_matrix, dtype=tf.float32)
            # 矩阵形状: (num_genes, num_go_terms)
            go_dim = go_matrix.shape[1]
            print(f"DeepCOP: 集成 GO 特征矩阵，维度 {go_dim}")
            
        # Input Normalization (Crucial for stability)
        self.input_bn = layers.BatchNormalization(name='input_bn')
        
        input_dim = int(num_genes) + int(self.drug_dim) + int(go_dim)
        if self.original_architecture:
            self.hidden_dim = int(input_dim)
            act1 = "selu"
            act2 = "relu"
        else:
            self.hidden_dim = int(hidden_dim)
            act1 = "relu"
            act2 = "relu"
        
        # Layer 1: Dense -> BN -> ReLU -> Dropout
        self.dense1 = layers.Dense(self.hidden_dim, name='dense_1')
        self.bn1 = layers.BatchNormalization(name='bn_1')
        self.act1 = layers.Activation(act1, name='act_1')
        self.drop1 = layers.Dropout(dropout, name='drop_1')
        
        # Layer 2: Dense -> BN -> ReLU -> Dropout
        self.dense2 = layers.Dense(self.hidden_dim, name='dense_2')
        self.bn2 = layers.BatchNormalization(name='bn_2')
        self.act2 = layers.Activation(act2, name='act_2')
        self.drop2 = layers.Dropout(dropout, name='drop_2')
        
        # Output Layer: Dense(num_genes) -> Linear
        # Note: Original DeepCOP used Softmax for classification. 
        # We use Linear for regression to match RFA-GNN task.
        self.dense_out = layers.Dense(num_genes, name='output')
        if self.predict_uncertainty:
            self.dense_logvar = layers.Dense(num_genes, name='output_logvar')
        
    def call(self, inputs, training=False):
        """
        Forward pass
        inputs: [ctl_expr, drug_fp]
          - ctl_expr: (Batch, num_genes)
          - drug_fp: (Batch, drug_dim)
        """
        ctl_expr, drug_fp = inputs
        
        # 1. Feature Fusion
        # 计算 Pathway Activity: (B, N_Genes) @ (N_Genes, N_GO) -> (B, N_GO)
        # 这一步将基因层面的表达值聚合为通路层面的活性值
        features_list = [ctl_expr]
        
        if self.go_matrix is not None:
            pathway_act = tf.matmul(ctl_expr, self.go_matrix)
            features_list.append(pathway_act)
            
        features_list.append(drug_fp)
        
        # Concatenate: (Batch, Total_Dim)
        x = tf.concat(features_list, axis=-1)
        
        # 0. Input BN
        x = self.input_bn(x, training=training)
        
        # 2. Layer 1
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.drop1(x, training=training)
        
        # 3. Layer 2
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.drop2(x, training=training)
        
        # 4. Output
        out = self.dense_out(x)
        if self.predict_uncertainty:
            logvar = self.dense_logvar(x)
        
        # 5. Residual Connection (Optional but Recommended)
        if self.use_residual:
            mean_out = ctl_expr + out
        else:
            mean_out = out
        if self.predict_uncertainty:
            return tf.stack([mean_out, logvar], axis=-1)
        return mean_out

    def get_config(self):
        config = super(DeepCOP, self).get_config()
        config.update({
            "num_genes": self.num_genes,
            "drug_dim": self.drug_dim,
            "dropout": self.dropout_rate,
            "use_residual": self.use_residual,
            "original_architecture": self.original_architecture,
            "hidden_dim": self.hidden_dim,
            "predict_uncertainty": self.predict_uncertainty,
        })
        return config
