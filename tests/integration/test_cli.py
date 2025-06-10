"""
CLI Integration Tests
=====================

Test command-line interface functionality.
"""

import pytest
import subprocess
import sys
from pathlib import Path
import json
import yaml

from click.testing import CliRunner

from retfound.cli.main import cli
from retfound.cli.commands.train import train_command
from retfound.cli.commands.evaluate import evaluate_command
from retfound.cli.commands.export import export_command
from retfound.cli.commands.predict import predict_command


@pytest.mark.integration
class TestCLIBasic:
    """Test basic CLI functionality"""
    
    def test_cli_help(self):
        """Test CLI help command"""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'RETFound Training CLI' in result.output
        assert 'train' in result.output
        assert 'evaluate' in result.output
        assert 'export' in result.output
        assert 'predict' in result.output
    
    def test_cli_version(self):
        """Test CLI version command"""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert 'retfound' in result.output.lower()
    
    def test_train_help(self):
        """Test train command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['train', '--help'])
        
        assert result.exit_code == 0
        assert '--config' in result.output
        assert '--epochs' in result.output
        assert '--batch-size' in result.output


@pytest.mark.integration
class TestTrainCommand:
    """Test train command functionality"""
    
    def test_train_basic(self, temp_dataset_dir, temp_output_dir, minimal_config_file):
        """Test basic training via CLI"""
        runner = CliRunner()
        
        # Create minimal training command
        result = runner.invoke(cli, [
            'train',
            '--config', str(minimal_config_file),
            '--dataset', str(temp_dataset_dir),
            '--output', str(temp_output_dir),
            '--epochs', '2',
            '--debug'
        ])
        
        # Should complete successfully
        assert result.exit_code == 0
        assert 'Training completed' in result.output
        
        # Check outputs created
        assert (temp_output_dir / 'checkpoints').exists()
        assert (temp_output_dir / 'logs').exists()
    
    def test_train_resume(self, temp_dataset_dir, temp_output_dir, sample_checkpoint):
        """Test resuming training via CLI"""
        runner = CliRunner()
        
        result = runner.invoke(cli, [
            'train',
            '--dataset', str(temp_dataset_dir),
            '--output', str(temp_output_dir),
            '--resume', str(sample_checkpoint),
            '--epochs', '2',
            '--debug'
        ])
        
        assert result.exit_code == 0
        assert 'Resuming from checkpoint' in result.output
    
    def test_train_with_wandb(self, temp_dataset_dir, temp_output_dir, monkeypatch):
        """Test training with W&B integration"""
        # Mock wandb to avoid actual logging
        monkeypatch.setattr('wandb.init', lambda **kwargs: None)
        monkeypatch.setattr('wandb.log', lambda **kwargs: None)
        monkeypatch.setattr('wandb.finish', lambda: None)
        
        runner = CliRunner()
        
        result = runner.invoke(cli, [
            'train',
            '--dataset', str(temp_dataset_dir),
            '--output', str(temp_output_dir),
            '--wandb-project', 'test-project',
            '--wandb-entity', 'test-entity',
            '--epochs', '1',
            '--debug'
        ])
        
        assert result.exit_code == 0
    
    def test_train_invalid_config(self, temp_output_dir):
        """Test training with invalid configuration"""
        runner = CliRunner()
        
        # Create invalid config
        invalid_config = temp_output_dir / 'invalid.yaml'
        with open(invalid_config, 'w') as f:
            f.write("invalid: yaml: content:")
        
        result = runner.invoke(cli, [
            'train',
            '--config', str(invalid_config),
            '--debug'
        ])
        
        assert result.exit_code != 0
        assert 'error' in result.output.lower()


@pytest.mark.integration
class TestEvaluateCommand:
    """Test evaluate command functionality"""
    
    def test_evaluate_basic(self, sample_checkpoint, temp_dataset_dir, temp_output_dir):
        """Test basic evaluation via CLI"""
        runner = CliRunner()
        
        result = runner.invoke(cli, [
            'evaluate',
            '--checkpoint', str(sample_checkpoint),
            '--dataset', str(temp_dataset_dir),
            '--output', str(temp_output_dir)
        ])
        
        assert result.exit_code == 0
        assert 'Evaluation Results' in result.output
        assert 'Accuracy' in result.output
        
        # Check report generated
        assert (temp_output_dir / 'evaluation_report.json').exists()
    
    def test_evaluate_with_tta(self, sample_checkpoint, temp_dataset_dir, temp_output_dir):
        """Test evaluation with test-time augmentation"""
        runner = CliRunner()
        
        result = runner.invoke(cli, [
            'evaluate',
            '--checkpoint', str(sample_checkpoint),
            '--dataset', str(temp_dataset_dir),
            '--output', str(temp_output_dir),
            '--tta',
            '--tta-transforms', '3'
        ])
        
        assert result.exit_code == 0
        assert 'TTA enabled' in result.output
    
    def test_evaluate_clinical_report(self, sample_checkpoint, temp_dataset_dir, temp_output_dir):
        """Test clinical report generation"""
        runner = CliRunner()
        
        result = runner.invoke(cli, [
            'evaluate',
            '--checkpoint', str(sample_checkpoint),
            '--dataset', str(temp_dataset_dir),
            '--output', str(temp_output_dir),
            '--clinical-report'
        ])
        
        assert result.exit_code == 0
        assert (temp_output_dir / 'clinical_evaluation.txt').exists()
        
        # Check report content
        with open(temp_output_dir / 'clinical_evaluation.txt') as f:
            report = f.read()
            assert 'CLINICAL EVALUATION REPORT' in report
            assert 'Sensitivity' in report
            assert 'Specificity' in report


@pytest.mark.integration
class TestExportCommand:
    """Test export command functionality"""
    
    def test_export_torchscript(self, sample_checkpoint, temp_output_dir):
        """Test TorchScript export via CLI"""
        runner = CliRunner()
        
        result = runner.invoke(cli, [
            'export',
            '--checkpoint', str(sample_checkpoint),
            '--output', str(temp_output_dir),
            '--format', 'torchscript'
        ])
        
        assert result.exit_code == 0
        assert 'Export completed' in result.output
        assert (temp_output_dir / 'model.pt').exists()
    
    def test_export_onnx(self, sample_checkpoint, temp_output_dir):
        """Test ONNX export via CLI"""
        pytest.importorskip("onnx")
        
        runner = CliRunner()
        
        result = runner.invoke(cli, [
            'export',
            '--checkpoint', str(sample_checkpoint),
            '--output', str(temp_output_dir),
            '--format', 'onnx',
            '--opset', '14'
        ])
        
        assert result.exit_code == 0
        assert (temp_output_dir / 'model.onnx').exists()
    
    def test_export_all_formats(self, sample_checkpoint, temp_output_dir):
        """Test exporting all formats"""
        runner = CliRunner()
        
        result = runner.invoke(cli, [
            'export',
            '--checkpoint', str(sample_checkpoint),
            '--output', str(temp_output_dir),
            '--all-formats'
        ])
        
        assert result.exit_code == 0
        assert 'Exporting to all formats' in result.output
    
    def test_export_with_optimization(self, sample_checkpoint, temp_output_dir):
        """Test export with optimization"""
        runner = CliRunner()
        
        result = runner.invoke(cli, [
            'export',
            '--checkpoint', str(sample_checkpoint),
            '--output', str(temp_output_dir),
            '--format', 'torchscript',
            '--optimize',
            '--quantize'
        ])
        
        assert result.exit_code == 0
        assert 'Optimization enabled' in result.output


@pytest.mark.integration
class TestPredictCommand:
    """Test predict command functionality"""
    
    def test_predict_single_image(self, sample_checkpoint, sample_image_files, temp_output_dir):
        """Test single image prediction"""
        runner = CliRunner()
        
        result = runner.invoke(cli, [
            'predict',
            '--checkpoint', str(sample_checkpoint),
            '--input', str(sample_image_files[0]),
            '--output', str(temp_output_dir / 'prediction.json')
        ])
        
        assert result.exit_code == 0
        assert 'Prediction' in result.output
        
        # Check output file
        assert (temp_output_dir / 'prediction.json').exists()
        
        with open(temp_output_dir / 'prediction.json') as f:
            pred = json.load(f)
            assert 'predicted_class' in pred
            assert 'confidence' in pred
            assert 'probabilities' in pred
    
    def test_predict_batch(self, sample_checkpoint, temp_dataset_dir, temp_output_dir):
        """Test batch prediction on directory"""
        runner = CliRunner()
        
        result = runner.invoke(cli, [
            'predict',
            '--checkpoint', str(sample_checkpoint),
            '--input', str(temp_dataset_dir / 'test'),
            '--output', str(temp_output_dir / 'predictions.csv'),
            '--batch-size', '4'
        ])
        
        assert result.exit_code == 0
        assert 'Processing' in result.output
        assert (temp_output_dir / 'predictions.csv').exists()
    
    def test_predict_with_tta(self, sample_checkpoint, sample_image_files, temp_output_dir):
        """Test prediction with test-time augmentation"""
        runner = CliRunner()
        
        result = runner.invoke(cli, [
            'predict',
            '--checkpoint', str(sample_checkpoint),
            '--input', str(sample_image_files[0]),
            '--output', str(temp_output_dir / 'prediction_tta.json'),
            '--tta'
        ])
        
        assert result.exit_code == 0
        assert 'TTA enabled' in result.output
    
    def test_predict_visualization(self, sample_checkpoint, sample_image_files, temp_output_dir):
        """Test prediction with visualization"""
        runner = CliRunner()
        
        result = runner.invoke(cli, [
            'predict',
            '--checkpoint', str(sample_checkpoint),
            '--input', str(sample_image_files[0]),
            '--output', str(temp_output_dir / 'prediction.json'),
            '--visualize',
            '--viz-output', str(temp_output_dir / 'visualization.png')
        ])
        
        assert result.exit_code == 0
        assert (temp_output_dir / 'visualization.png').exists()


@pytest.mark.integration
class TestCLIPipeline:
    """Test complete CLI pipelines"""
    
    def test_train_evaluate_export_pipeline(self, temp_dataset_dir, temp_output_dir, minimal_config_file):
        """Test complete pipeline from training to export"""
        runner = CliRunner()
        
        # Step 1: Train
        result = runner.invoke(cli, [
            'train',
            '--config', str(minimal_config_file),
            '--dataset', str(temp_dataset_dir),
            '--output', str(temp_output_dir),
            '--epochs', '2',
            '--debug'
        ])
        assert result.exit_code == 0
        
        # Find checkpoint
        checkpoint = list((temp_output_dir / 'checkpoints').glob('*.pth'))[0]
        
        # Step 2: Evaluate
        result = runner.invoke(cli, [
            'evaluate',
            '--checkpoint', str(checkpoint),
            '--dataset', str(temp_dataset_dir),
            '--output', str(temp_output_dir)
        ])
        assert result.exit_code == 0
        
        # Step 3: Export
        result = runner.invoke(cli, [
            'export',
            '--checkpoint', str(checkpoint),
            '--output', str(temp_output_dir),
            '--format', 'torchscript'
        ])
        assert result.exit_code == 0
        
        # Step 4: Predict
        test_images = list((temp_dataset_dir / 'test').glob('*/*.jpg'))
        if test_images:
            result = runner.invoke(cli, [
                'predict',
                '--checkpoint', str(checkpoint),
                '--input', str(test_images[0]),
                '--output', str(temp_output_dir / 'final_prediction.json')
            ])
            assert result.exit_code == 0


@pytest.mark.integration
class TestCLIErrorHandling:
    """Test CLI error handling"""
    
    def test_missing_required_args(self):
        """Test handling of missing required arguments"""
        runner = CliRunner()
        
        # Train without dataset
        result = runner.invoke(cli, ['train'])
        assert result.exit_code != 0
        assert 'Missing' in result.output or 'required' in result.output
    
    def test_invalid_checkpoint(self, temp_output_dir):
        """Test handling of invalid checkpoint"""
        runner = CliRunner()
        
        invalid_checkpoint = temp_output_dir / 'invalid.pth'
        invalid_checkpoint.write_text('invalid data')
        
        result = runner.invoke(cli, [
            'evaluate',
            '--checkpoint', str(invalid_checkpoint),
            '--dataset', str(temp_output_dir)
        ])
        
        assert result.exit_code != 0
        assert 'error' in result.output.lower()
    
    def test_keyboard_interrupt_handling(self, temp_dataset_dir, temp_output_dir, monkeypatch):
        """Test handling of keyboard interrupt"""
        runner = CliRunner()
        
        # Mock to raise KeyboardInterrupt during training
        def mock_train(*args, **kwargs):
            raise KeyboardInterrupt()
        
        monkeypatch.setattr('retfound.cli.commands.train.train_model', mock_train)
        
        result = runner.invoke(cli, [
            'train',
            '--dataset', str(temp_dataset_dir),
            '--output', str(temp_output_dir),
            '--epochs', '1'
        ])
        
        assert 'Interrupted by user' in result.output


@pytest.mark.integration
class TestCLISubprocess:
    """Test CLI via subprocess (real command line)"""
    
    def test_cli_as_module(self):
        """Test running CLI as Python module"""
        result = subprocess.run(
            [sys.executable, '-m', 'retfound', '--help'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'RETFound' in result.stdout
    
    def test_cli_script_entry_point(self):
        """Test CLI script entry point"""
        # This would test the installed script, skip if not installed
        try:
            result = subprocess.run(
                ['retfound', '--help'],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
            assert 'RETFound' in result.stdout
        except FileNotFoundError:
            pytest.skip("retfound script not installed")