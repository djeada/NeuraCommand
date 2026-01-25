"""
Tests for the CLI command handler.

Tests the redesigned command handler with proper subparser support
and both new command syntax and legacy --action syntax.
"""

import sys
import tempfile
import os
from io import StringIO
from unittest.mock import patch

import pytest

from app.commands.command_handler import CommandHandler


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
            '--network-architecture-json', str(arch_file),
            '--model-pkl', str(tmp_path / 'model.pkl')
        ]):
            handler = CommandHandler()
            assert handler.args.command == 'create'
            assert handler.args.network_architecture_json == str(arch_file)
            assert handler.args.model_pkl == str(tmp_path / 'model.pkl')

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
        assert 'required' in captured.err.lower() or 'network-architecture-json' in captured.err


class TestCommandDescriptions:
    """Tests for command descriptions."""

    def test_all_commands_have_descriptions(self):
        """Test that all commands have descriptions in the handler."""
        # Verify all command classes are in command_map
        expected_commands = ['create', 'train', 'predict', 'info', 'export', 'analyze', 'visualize']
        for cmd in expected_commands:
            assert cmd in CommandHandler.COMMAND_DESCRIPTIONS
