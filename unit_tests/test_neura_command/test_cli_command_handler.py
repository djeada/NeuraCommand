"""
Tests for the CLI command handler.

Tests the redesigned command handler with proper subparser support
and both new command syntax and legacy --action syntax.
"""

import sys
import tempfile
import os
import runpy
from io import StringIO
from unittest.mock import patch

import pytest

from app.commands.command_handler import CommandHandler
from neura_command.cli import main as cli_main


class TestCommandHandlerHelp:
    """Tests for help display functionality."""

    def test_main_help_shows_commands(self, capsys):
        """Test that main help shows all available commands."""
        with patch.object(sys, 'argv', ['neura_command', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                CommandHandler()
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert 'create' in captured.out
        assert 'train' in captured.out
        assert 'predict' in captured.out
        assert 'info' in captured.out
        assert 'export' in captured.out
        assert 'analyze' in captured.out
        assert 'visualize' in captured.out

    def test_version_flag_short(self, capsys):
        """Test that -v flag shows version."""
        with patch.object(sys, 'argv', ['neura_command', '-v']):
            with pytest.raises(SystemExit) as exc_info:
                CommandHandler()
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert 'neura_command' in captured.out

    def test_version_flag_long(self, capsys):
        """Test that --version flag shows version."""
        with patch.object(sys, 'argv', ['neura_command', '--version']):
            with pytest.raises(SystemExit) as exc_info:
                CommandHandler()
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert 'neura_command' in captured.out

    def test_subcommand_help_create(self, capsys):
        """Test that 'create --help' shows create-specific help."""
        with patch.object(sys, 'argv', ['neura_command', 'create', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                CommandHandler()
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert '--network-architecture-json' in captured.out
        assert '--model-pkl' in captured.out

    def test_subcommand_help_train(self, capsys):
        """Test that 'train --help' shows train-specific help."""
        with patch.object(sys, 'argv', ['neura_command', 'train', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                CommandHandler()
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert '--model-pkl' in captured.out
        assert '--features-csv' in captured.out
        assert '--targets-csv' in captured.out
        assert '--epochs' in captured.out


class TestCommandHandlerParsing:
    """Tests for command argument parsing."""

    def test_new_syntax_create(self, tmp_path):
        """Test new syntax: 'neura_command create ...'"""
        # Create a temporary JSON file
        arch_file = tmp_path / "arch.json"
        arch_file.write_text('{"layers": []}')
        
        with patch.object(sys, 'argv', [
            'neura_command', 'create',
            '--architecture', str(arch_file),
            '--output', str(tmp_path / 'model')
        ]):
            handler = CommandHandler()
            assert handler.args.command == 'create'
            assert handler.args.architecture == str(arch_file)
            assert handler.args.output == str(tmp_path / 'model')

    def test_new_syntax_info(self):
        """Test new syntax: 'neura_command info ...'"""
        with patch.object(sys, 'argv', [
            'neura_command', 'info',
            '--model', '/path/to/model',
            '--format', 'json'
        ]):
            handler = CommandHandler()
            assert handler.args.command == 'info'
            assert handler.args.model == '/path/to/model'
            assert handler.args.format == 'json'

    def test_new_syntax_analyze(self):
        """Test new syntax: 'neura_command analyze ...'"""
        with patch.object(sys, 'argv', [
            'neura_command', 'analyze',
            '--model', '/path/to/model',
            '--analysis-type', 'architecture',
            '--format', 'text'
        ]):
            handler = CommandHandler()
            assert handler.args.command == 'analyze'
            assert handler.args.model == '/path/to/model'
            assert handler.args.analysis_type == 'architecture'


class TestCommandHandlerErrors:
    """Tests for error handling in command handler."""

    def test_no_command_shows_help(self, capsys):
        """Test that no command shows help and exits with error."""
        with patch.object(sys, 'argv', ['neura_command']):
            handler = CommandHandler()
            with pytest.raises(SystemExit) as exc_info:
                handler.execute_command()
            assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert 'Please specify a command' in captured.out or 'error' in captured.out.lower()

    def test_missing_required_args_create(self, capsys):
        """Test that missing required args for create shows error."""
        with patch.object(sys, 'argv', ['neura_command', 'create']):
            with pytest.raises(SystemExit) as exc_info:
                CommandHandler()
            assert exc_info.value.code == 2

        captured = capsys.readouterr()
        assert 'required' in captured.err.lower() or 'architecture' in captured.err


class TestCommandDescriptions:
    """Tests for command descriptions."""

    def test_all_commands_have_descriptions(self):
        """Test that all commands have descriptions in the handler."""
        # Verify all command classes are in command_map
        expected_commands = ['create', 'train', 'predict', 'info', 'export', 'analyze', 'visualize']
        for cmd in expected_commands:
            assert cmd in CommandHandler.COMMAND_DESCRIPTIONS


class TestNewCLIFeatures:
    """Tests for the new CLI features."""

    def test_create_with_verbose_flag(self, tmp_path):
        """Test that create command accepts verbose flag."""
        arch_file = tmp_path / "arch.json"
        arch_file.write_text('{"layers": []}')
        
        with patch.object(sys, 'argv', [
            'neura_command', 'create',
            '--architecture', str(arch_file),
            '--verbose'
        ]):
            handler = CommandHandler()
            assert handler.args.verbose is True

    def test_create_with_quiet_flag(self, tmp_path):
        """Test that create command accepts quiet flag."""
        arch_file = tmp_path / "arch.json"
        arch_file.write_text('{"layers": []}')
        
        with patch.object(sys, 'argv', [
            'neura_command', 'create',
            '--architecture', str(arch_file),
            '--quiet'
        ]):
            handler = CommandHandler()
            assert handler.args.quiet is True

    def test_create_with_validate_flag(self, tmp_path):
        """Test that create command accepts validate flag."""
        arch_file = tmp_path / "arch.json"
        arch_file.write_text('{"layers": []}')
        
        with patch.object(sys, 'argv', [
            'neura_command', 'create',
            '--architecture', str(arch_file),
            '--validate'
        ]):
            handler = CommandHandler()
            assert handler.args.validate is True

    def test_train_with_batch_size(self, tmp_path):
        """Test train command with batch size option."""
        # Create mock files
        model = tmp_path / "model"
        features = tmp_path / "features.csv"
        targets = tmp_path / "targets.csv"
        features.write_text("1.0,2.0\n3.0,4.0")
        targets.write_text("0.0\n1.0")
        
        with patch.object(sys, 'argv', [
            'neura_command', 'train',
            '--model', str(model),
            '--features-csv', str(features),
            '--targets-csv', str(targets),
            '--batch-size', '16',
            '--learning-rate', '0.01'
        ]):
            handler = CommandHandler()
            assert handler.args.batch_size == 16
            assert handler.args.learning_rate == 0.01

    def test_train_with_early_stopping(self, tmp_path):
        """Test train command with early stopping option."""
        model = tmp_path / "model"
        features = tmp_path / "features.csv"
        targets = tmp_path / "targets.csv"
        features.write_text("1.0,2.0\n3.0,4.0")
        targets.write_text("0.0\n1.0")
        
        with patch.object(sys, 'argv', [
            'neura_command', 'train',
            '--model', str(model),
            '--features-csv', str(features),
            '--targets-csv', str(targets),
            '--early-stopping', '10'
        ]):
            handler = CommandHandler()
            assert handler.args.early_stopping == 10

    def test_predict_with_confidence(self, tmp_path):
        """Test predict command with confidence options."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("1.0,2.0\n3.0,4.0")
        
        with patch.object(sys, 'argv', [
            'neura_command', 'predict',
            '--model', '/some/model',
            '--input-csv', str(input_csv),
            '--show-confidence',
            '--top-k', '3',
            '--threshold', '0.7'
        ]):
            handler = CommandHandler()
            assert handler.args.show_confidence is True
            assert handler.args.top_k == 3
            assert handler.args.threshold == 0.7

    def test_predict_with_json_output(self, tmp_path):
        """Test predict command with JSON output mode."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("1.0,2.0\n3.0,4.0")
        
        with patch.object(sys, 'argv', [
            'neura_command', 'predict',
            '--model', '/some/model',
            '--input-csv', str(input_csv),
            '--output-mode', 'json',
            '--precision', '4'
        ]):
            handler = CommandHandler()
            assert handler.args.output_mode == 'json'
            assert handler.args.precision == 4

    def test_backward_compatible_arg_names(self, tmp_path):
        """Test that legacy argument names still work."""
        arch_file = tmp_path / "arch.json"
        arch_file.write_text('{"layers": []}')
        
        # Test --network-architecture-json (legacy name for --architecture)
        with patch.object(sys, 'argv', [
            'neura_command', 'create',
            '--network-architecture-json', str(arch_file),
        ]):
            handler = CommandHandler()
            assert handler.args.architecture == str(arch_file)


class TestConsoleEntrypoints:
    def test_cli_entrypoint_uses_supplied_argv(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["--help"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "commands" in captured.out

    def test_module_entrypoint_runs(self, capsys):
        with patch.object(sys, "argv", ["neura_command", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_module("neura_command.__main__", run_name="__main__")
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "neura_command" in captured.out
