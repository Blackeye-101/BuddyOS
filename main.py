"""
BuddyOS Main Entry Point

Interactive CLI for chatting with Buddy.
Demonstrates persistent memory across sessions.
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure data directory exists
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

from core.discovery import discover_available_models, get_default_model
from core.database import create_database
from core.router import create_router
from agents.buddy import create_orchestrator


class BuddyOSCLI:
    """
    Interactive CLI for BuddyOS.
    
    Features:
    - Model selection from available models
    - Persistent conversations
    - User fact memory across sessions
    """
    
    def __init__(self):
        self.db = None
        self.router = None
        self.orchestrator = None
        self.current_conversation_id = None
        self.current_model_id = None
        self.available_models = []  # Store available models for switching
    
    async def initialize(self):
        """Initialize all BuddyOS components."""
        print("🤖 BuddyOS Initializing...")
        print()
        
        # Step 1: Discover available models (Wave 1)
        print("📡 Discovering available models...")
        self.available_models = discover_available_models(include_paid=True)
        
        if not self.available_models:
            print("⚠️  No API keys configured. Using fallback model.")
            default_model = get_default_model()
            self.available_models = [default_model]
        
        print(f"✅ Found {len(self.available_models)} available models")
        print()
        
        # Step 2: Initialize databases (Wave 2 - Step 2)
        print("💾 Initializing databases...")
        self.db = await create_database(
            sqlite_path="data/buddy.db",
            duckdb_path="data/knowledge.duckdb"
        )
        print("✅ SQLite and DuckDB initialized")
        print()
        
        # Step 3: Initialize router (Wave 2 - Step 3)
        print("🔀 Initializing router...")
        self.router = create_router()
        print("✅ Router ready with fallback chain")
        print()
        
        # Step 4: Initialize orchestrator (Wave 2 - Step 4)
        print("🧠 Initializing orchestrator...")
        self.orchestrator = create_orchestrator(self.router, self.db)
        print("✅ Orchestrator ready")
        print()
        
        # Display available models
        print("📋 Available Models:")
        for i, model in enumerate(self.available_models, 1):
            tier_icon = "🆓" if model.tier == "free" else "💰"
            print(f"  {i}. {tier_icon} {model.display_name} ({model.provider})")
        print()
        
        # Select default model - Gemini 2.5 Flash (stable)
        default_model = None
        for model in self.available_models:
            if model.model_id == "gemini/gemini-2.5-flash":  # ✅ Actual Google API name
                default_model = model
                break
        
        # If Gemini 2.5 Flash not available, use first model
        if not default_model:
            default_model = self.available_models[0]
        
        self.current_model_id = default_model.model_id
        print(f"🎯 Using default model: {default_model.display_name}")
        print()
        
        # Check for existing user facts
        facts = await self.db.get_user_facts(active_only=True)
        if facts:
            print(f"💡 Loaded {len(facts)} facts about you from previous sessions")
            print()
        
        print("=" * 60)
        print("🚀 BuddyOS Ready!")
        print("=" * 60)
        print()
    
    async def display_welcome(self):
        """Display welcome message and instructions."""
        print("Welcome to BuddyOS - Your Model-Agnostic AI Assistant")
        print()
        print("Commands:")
        print("  /help     - Show this help message")
        print("  /facts    - Show what Buddy knows about you")
        print("  /model    - Switch to a different model")
        print("  /new      - Start a new conversation")
        print("  /history  - Show recent conversations")
        print("  /exit     - Exit BuddyOS")
        print()
        print("Just type your message to chat with Buddy!")
        print()
    
    async def show_facts(self):
        """Display all user facts."""
        facts = await self.db.get_user_facts(active_only=True)
        
        if not facts:
            print("📝 No facts recorded yet. Share information about yourself!")
            print()
            return
        
        # Group by category
        from collections import defaultdict
        facts_by_category = defaultdict(list)
        
        for fact in facts:
            facts_by_category[fact.category].append(fact)
        
        print("📚 What Buddy Knows About You:")
        print()
        
        for category, category_facts in facts_by_category.items():
            print(f"**{category}:**")
            for fact in category_facts:
                confidence_icon = "✓" if fact.confidence >= 0.85 else "~"
                confidence_pct = int(fact.confidence * 100)
                print(f"  {confidence_icon} {fact.fact_text} ({confidence_pct}% confidence)")
            print()
    
    async def show_history(self):
        """Display recent conversations."""
        conversations = await self.db.list_conversations(limit=10)
        
        if not conversations:
            print("📝 No previous conversations")
            print()
            return
        
        print("📜 Recent Conversations:")
        print()
        
        for i, conv in enumerate(conversations, 1):
            # Get message count
            messages = await self.db.get_conversation_history(conv.id)
            print(f"{i}. {conv.title}")
            print(f"   Created: {conv.created_at}")
            print(f"   Messages: {len(messages)}")
            print()
    
    async def show_model_selection(self):
        """Display model selection menu and allow user to switch."""
        if not self.available_models:
            print("❌ No models available")
            print()
            return
        
        # Get current model info
        current_model_info = None
        for model in self.available_models:
            if model.model_id == self.current_model_id:
                current_model_info = model
                break
        
        print("🤖 Available Models:")
        print()
        
        for i, model in enumerate(self.available_models, 1):
            tier_icon = "🆓" if model.tier == "free" else "💰"
            current_marker = " ← Current" if model.model_id == self.current_model_id else ""
            print(f"  {i}. {tier_icon} {model.display_name} ({model.provider}){current_marker}")
        
        print()
        print("Enter model number to switch (or 'c' to cancel): ", end="")
        
        try:
            choice = input().strip().lower()
            
            if choice == 'c' or choice == '':
                print("Model selection cancelled")
                print()
                return
            
            # Try to parse as integer
            try:
                model_index = int(choice) - 1
                
                if 0 <= model_index < len(self.available_models):
                    selected_model = self.available_models[model_index]
                    old_model_name = current_model_info.display_name if current_model_info else "Unknown"
                    
                    self.current_model_id = selected_model.model_id
                    
                    print(f"✅ Switched from {old_model_name} to {selected_model.display_name}")
                    print()
                    
                    # Start a new conversation with the new model
                    if self.current_conversation_id:
                        print("💡 Starting new conversation with selected model...")
                        await self.start_new_conversation()
                else:
                    print(f"❌ Invalid selection. Please choose 1-{len(self.available_models)}")
                    print()
            except ValueError:
                print("❌ Invalid input. Please enter a number or 'c' to cancel")
                print()
        
        except Exception as e:
            print(f"❌ Error during model selection: {e}")
            print()
    
    async def start_new_conversation(self):
        """Start a new conversation."""
        print("✨ Starting new conversation...")
        
        self.current_conversation_id = await self.orchestrator.start_new_conversation(
            model_id=self.current_model_id
        )
        
        print(f"✅ New conversation started: {self.current_conversation_id[:8]}...")
        print()
    
    async def chat_loop(self):
        """Main interactive chat loop."""
        await self.display_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    command = user_input.lower()
                    
                    if command == "/exit":
                        print("\n👋 Goodbye! Your memories are saved.")
                        break
                    
                    elif command == "/help":
                        await self.display_welcome()
                        continue
                    
                    elif command == "/facts":
                        await self.show_facts()
                        continue
                    
                    elif command == "/model":
                        await self.show_model_selection()
                        continue
                    
                    elif command == "/new":
                        await self.start_new_conversation()
                        continue
                    
                    elif command == "/history":
                        await self.show_history()
                        continue
                    
                    else:
                        print(f"❓ Unknown command: {user_input}")
                        print("Type /help for available commands")
                        print()
                        continue
                
                # Process message through orchestrator
                print("Buddy: ", end="", flush=True)
                
                response = await self.orchestrator.process_message(
                    user_message=user_input,
                    conversation_id=self.current_conversation_id,
                    model_id=self.current_model_id
                )
                
                # Update conversation ID if new
                if not self.current_conversation_id:
                    self.current_conversation_id = response.conversation_id
                
                # Display response
                print(response.response)
                
                # Show fallback info if it occurred
                if response.fallback_occurred:
                    print(f"\n⚠️  Note: Fell back from {response.fallback_from} to {response.model_used}")
                
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Your memories are saved.")
                break
            
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type /exit to quit")
                print()
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.db:
            await self.db.close()
        print("✅ Resources cleaned up")


async def main():
    """
    Main entry point for BuddyOS.
    
    Verification Test:
    1. Run: python main.py
    2. Say: "I am a scientist"
    3. Exit
    4. Run: python main.py again
    5. Ask: "What do you know about me?"
    6. Expected: Buddy should mention you're a scientist
    """
    cli = BuddyOSCLI()
    
    try:
        # Initialize all components
        await cli.initialize()
        
        # Run chat loop
        await cli.chat_loop()
        
    finally:
        # Cleanup
        await cli.cleanup()


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 12):
        print("❌ Error: Python 3.12 or higher required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    # Run async main
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")