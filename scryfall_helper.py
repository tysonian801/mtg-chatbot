"""
Scryfall API helper functions for MTG Rules Assistant.
This module provides functions to fetch card information from Scryfall API.
"""

import scrython
from typing import Optional, Dict, Any
import time

def search_card(card_name: str) -> Optional[Dict[str, Any]]:
    """
    Search for a card by name using Scryfall API.
    
    Args:
        card_name: Name of the card to search for
        
    Returns:
        Dictionary containing card information or None if not found
    """
    try:
        # Use Scrython to search for the card
        card = scrython.cards.Named(fuzzy=card_name)
        
        # Extract relevant information
        card_info = {
            'name': card.name(),
            'mana_cost': card.mana_cost(),
            'type_line': card.type_line(),
            'oracle_text': card.oracle_text(),
            'power': card.power(),
            'toughness': card.toughness(),
            'rarity': card.rarity(),
            'set_name': card.set_name(),
            'image_uris': card.image_uris() if hasattr(card, 'image_uris') else None
        }
        
        return card_info
    
    except Exception as e:
        print(f"Error searching for card '{card_name}': {str(e)}")
        return None

def get_card_rules_text(card_name: str) -> Optional[str]:
    """
    Get the rules text (oracle text) for a specific card.
    
    Args:
        card_name: Name of the card
        
    Returns:
        Rules text as string or None if card not found
    """
    card_info = search_card(card_name)
    if card_info and card_info.get('oracle_text'):
        return card_info['oracle_text']
    return None

def format_card_info_for_context(card_name: str) -> Optional[str]:
    """
    Format card information for inclusion in GPT context.
    
    Args:
        card_name: Name of the card
        
    Returns:
        Formatted string with card information or None if not found
    """
    card_info = search_card(card_name)
    if not card_info:
        return None
    
    # Format the card information
    context = f"Card: {card_info['name']}\n"
    if card_info.get('mana_cost'):
        context += f"Mana Cost: {card_info['mana_cost']}\n"
    context += f"Type: {card_info['type_line']}\n"
    if card_info.get('oracle_text'):
        context += f"Rules Text: {card_info['oracle_text']}\n"
    if card_info.get('power') and card_info.get('toughness'):
        context += f"Power/Toughness: {card_info['power']}/{card_info['toughness']}\n"
    
    return context

def search_multiple_cards(card_names: list[str]) -> Dict[str, Optional[str]]:
    """
    Search for multiple cards and return their rules text.
    
    Args:
        card_names: List of card names to search for
        
    Returns:
        Dictionary mapping card names to their rules text (or None if not found)
    """
    results = {}
    
    for card_name in card_names:
        rules_text = get_card_rules_text(card_name)
        results[card_name] = rules_text
        
        # Be nice to the API - add a small delay between requests
        time.sleep(0.1)
    
    return results

def get_relevant_cards_context(card_names: list[str]) -> str:
    """
    Get formatted context for multiple cards.
    
    Args:
        card_names: List of card names
        
    Returns:
        Formatted string with all relevant card information
    """
    if not card_names:
        return ""
    
    context_parts = []
    for card_name in card_names:
        card_context = format_card_info_for_context(card_name)
        if card_context:
            context_parts.append(card_context)
    
    if context_parts:
        return "\n---\n".join(context_parts)
    
    return "" 