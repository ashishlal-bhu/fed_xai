import tensorflow as tf
import logging
from typing import Dict, Optional

logger = logging.getLogger('model_factory')

def create_model(input_dim: int, model_config: Optional[Dict] = None) -> tf.keras.Model:
    """
    Factory function to create consistent models for server and clients
    """
    config = {
        'hidden_sizes': [256, 128, 64],
        'dropout': 0.3,
        'learning_rate': 0.001,
        'l2_reg': 0.01
    }
    if model_config:
        config.update(model_config)
    
    logger.info("Creating model with configuration:")
    for key, value in config.items():
        logger.info(f"- {key}: {value}")
    
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    
    # Create hidden layers
    for units in config['hidden_sizes']:
        x = tf.keras.layers.Dense(
            units,
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(config['l2_reg'])
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(config['dropout'])(x)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model