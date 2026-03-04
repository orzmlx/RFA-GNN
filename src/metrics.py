def masked_combined_loss(y_true, y_pred):
        # 1. Masking for MSE
        y_true_m = y_true * loss_mask
        y_pred_m = y_pred * loss_mask
        
        # Sum of squared errors / Number of valid genes
        valid_count = tf.reduce_sum(loss_mask)
        mse = tf.reduce_sum(tf.square(y_true_m - y_pred_m)) / (valid_count * tf.cast(tf.shape(y_true)[0], tf.float32))
        
        # 2. Masking for PCC (Extract valid columns)
        # tf.boolean_mask returns flattened tensor if mask is 1D? 
        # We need to gather columns.
        valid_indices = tf.where(loss_mask[0] > 0)[:, 0]
        y_t_valid = tf.gather(y_true, valid_indices, axis=1)
        y_p_valid = tf.gather(y_pred, valid_indices, axis=1)
        
        pcc = pcc_loss(y_t_valid, y_p_valid)
        
        return mse + 5.0 * pcc