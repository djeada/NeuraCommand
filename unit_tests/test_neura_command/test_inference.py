"""
Tests for inference runtime and quantization.
"""

import numpy as np
import pytest
import tempfile
import os

from neura_command import Sequential, Dense, ReLU, Sigmoid, Dropout, BatchNorm
from neura_command.inference import InferenceModel, quantize_model, QuantizationConfig
from neura_command.inference.quantization import get_quantization_stats


class TestInferenceModel:
    def test_from_sequential(self):
        model = Sequential([Dense(10, 5), ReLU(), Dense(5, 2)])
        inference = InferenceModel.from_sequential(model)
        
        assert len(inference.layers) == 3
    
    def test_predict(self):
        model = Sequential([Dense(10, 5), ReLU(), Dense(5, 2)])
        inference = InferenceModel.from_sequential(model)
        
        X = np.random.randn(20, 10)
        pred = inference.predict(X)
        
        assert pred.shape == (20, 2)
    
    def test_callable(self):
        model = Sequential([Dense(10, 5)])
        inference = InferenceModel.from_sequential(model)
        
        X = np.random.randn(5, 10)
        pred1 = inference.predict(X)
        pred2 = inference(X)
        
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_predictions_match_original(self):
        np.random.seed(42)
        model = Sequential([Dense(10, 5), ReLU(), Dense(5, 2), Sigmoid()])
        
        X = np.random.randn(20, 10)
        
        # Original model in eval mode
        for layer in model.layers:
            if hasattr(layer, 'eval_mode'):
                layer.eval_mode()
        original_pred = model.predict(X)
        
        # Inference model
        inference = InferenceModel.from_sequential(model)
        inference_pred = inference.predict(X)
        
        np.testing.assert_array_almost_equal(original_pred, inference_pred)
    
    def test_from_file(self):
        model = Sequential([Dense(10, 5), ReLU()])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model')
            model.save(path)
            
            inference = InferenceModel.from_file(path)
            
            X = np.random.randn(5, 10)
            pred = inference.predict(X)
            
            assert pred.shape == (5, 5)
    
    def test_memory_bytes(self):
        model = Sequential([Dense(100, 50), Dense(50, 10)])
        inference = InferenceModel.from_sequential(model)
        
        # Should report some memory usage
        assert inference.memory_bytes > 0
    
    def test_info(self):
        model = Sequential([Dense(10, 5)])
        inference = InferenceModel.from_sequential(model)
        
        info = inference.info()
        assert 'InferenceModel' in info
        assert 'Layers' in info


class TestQuantization:
    def test_float16_quantization(self):
        model = Sequential([Dense(10, 5), ReLU()])
        
        layers = quantize_model(model, QuantizationConfig(dtype='float16'))
        
        assert len(layers) == 2
        assert layers[0].dtype == 'float16'
    
    def test_int8_quantization(self):
        model = Sequential([Dense(10, 5), ReLU()])
        
        layers = quantize_model(model, QuantizationConfig(dtype='int8'))
        
        assert len(layers) == 2
        assert layers[0].dtype == 'int8'
        assert layers[0].weights.dtype == np.int8
    
    def test_float16_inference(self):
        np.random.seed(42)
        model = Sequential([Dense(10, 5), ReLU(), Dense(5, 2)])
        
        X = np.random.randn(20, 10)
        original_pred = model.predict(X)
        
        inference = InferenceModel.from_sequential(model, quantize='float16')
        quant_pred = inference.predict(X)
        
        # Float16 should be very close to original
        np.testing.assert_array_almost_equal(original_pred, quant_pred, decimal=2)
    
    def test_int8_inference(self):
        np.random.seed(42)
        model = Sequential([Dense(10, 5), ReLU(), Dense(5, 2)])
        
        X = np.random.randn(20, 10)
        original_pred = model.predict(X)
        
        inference = InferenceModel.from_sequential(model, quantize='int8')
        quant_pred = inference.predict(X)
        
        # Int8 has more error but should be reasonable
        np.testing.assert_array_almost_equal(original_pred, quant_pred, decimal=1)
    
    def test_float16_memory_reduction(self):
        model = Sequential([Dense(100, 50), Dense(50, 10)])
        
        inference_fp32 = InferenceModel.from_sequential(model)
        inference_fp16 = InferenceModel.from_sequential(model, quantize='float16')
        
        # Float16 should be about 50% smaller
        ratio = inference_fp16.memory_bytes / inference_fp32.memory_bytes
        assert ratio < 0.6  # Allow some tolerance
    
    def test_int8_memory_reduction(self):
        model = Sequential([Dense(100, 50), Dense(50, 10)])
        
        inference_fp32 = InferenceModel.from_sequential(model)
        inference_int8 = InferenceModel.from_sequential(model, quantize='int8')
        
        # Int8 should be about 25% of original (for weights)
        ratio = inference_int8.memory_bytes / inference_fp32.memory_bytes
        assert ratio < 0.5


class TestQuantizationStats:
    def test_get_stats(self):
        model = Sequential([Dense(100, 50), Dense(50, 10)])
        layers = quantize_model(model, QuantizationConfig(dtype='float16'))
        
        stats = get_quantization_stats(model, layers)
        
        assert 'original_bytes' in stats
        assert 'quantized_bytes' in stats
        assert 'compression_ratio' in stats
        assert stats['compression_ratio'] > 1.5  # Should compress


class TestLayerQuantization:
    def test_dense_quantized(self):
        model = Sequential([Dense(10, 5, use_bias=True)])
        layers = quantize_model(model, QuantizationConfig(dtype='float16'))
        
        assert layers[0].use_bias is True
        assert layers[0].weights.dtype == np.float16
        assert layers[0].bias.dtype == np.float16
    
    def test_dropout_quantized(self):
        model = Sequential([Dense(10, 5), Dropout(0.5)])
        layers = quantize_model(model, QuantizationConfig(dtype='float16'))
        
        # Dropout should pass through in inference
        X = np.random.randn(10, 5)
        out = layers[1].forward(X)
        np.testing.assert_array_equal(X, out)
    
    def test_batchnorm_quantized(self):
        model = Sequential([Dense(10, 5), BatchNorm(5)])
        
        # Do forward pass to set running stats
        X = np.random.randn(32, 10)
        model.forward(X, training=True)
        
        layers = quantize_model(model, QuantizationConfig(dtype='float16'))
        
        assert layers[1].dtype == 'float16'
        assert layers[1].gamma.dtype == np.float16
    
    def test_all_activations_quantized(self):
        from neura_command import Softmax, Sigmoid
        from neura_command.layers.activation import Tanh
        
        model = Sequential([
            Dense(10, 5),
            ReLU(),
            Sigmoid(),
            Softmax(),
            Tanh(),
        ])
        
        layers = quantize_model(model, QuantizationConfig(dtype='float16'))
        
        assert len(layers) == 5
        
        # Test each activation works
        X = np.random.randn(5, 5)
        for i, layer in enumerate(layers[1:], 1):
            out = layer.forward(X)
            assert out.shape == X.shape


class TestInferenceFromFile:
    def test_quantized_load(self):
        model = Sequential([Dense(10, 5), ReLU()])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model')
            model.save(path)
            
            # Load with float16 quantization
            inference = InferenceModel.from_file(path, quantize='float16')
            
            assert inference.metadata.get('quantized') == 'float16'
