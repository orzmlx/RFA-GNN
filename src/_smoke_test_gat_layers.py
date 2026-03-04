import tensorflow as tf

from base_gnn import BaseLineGAT


def main():
    b = 2
    n = 64
    num_cells = 3
    adj = tf.eye(n)
    ctl = tf.random.normal((b, n, 2))
    drug = tf.random.uniform((b, n))
    cell = tf.random.uniform((b,), minval=0, maxval=num_cells, dtype=tf.int32)

    m = BaseLineGAT(
        num_genes=n,
        num_cells=num_cells,
        hidden_dim=32,
        num_heads=4,
        dropout=0.1,
        attention_layer_number=3,
        use_residual=False,
        use_drug_embedding=False,
    )
    out = m([adj, ctl, drug, cell])
    assert out.shape == (b, n), out.shape
    print("OK", out.shape)


if __name__ == "__main__":
    main()

