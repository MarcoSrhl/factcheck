#!/usr/bin/env python
"""Test script to verify factcheck package installation."""

def test_imports():
    """Test that all main components can be imported."""
    try:
        from factcheck import (
            FactChecker,
            FactClassifier,
            TripletExtractor,
            EntityLinker,
            KnowledgeQuery,
            LABEL_MAP,
            LABEL_TO_ID,
            NUM_LABELS,
        )
        print("✅ All imports successful!")
        print(f"   - FactChecker: {FactChecker}")
        print(f"   - FactClassifier: {FactClassifier}")
        print(f"   - TripletExtractor: {TripletExtractor}")
        print(f"   - EntityLinker: {EntityLinker}")
        print(f"   - KnowledgeQuery: {KnowledgeQuery}")
        print(f"   - Labels: {LABEL_MAP}")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without loading models."""
    try:
        from factcheck import TripletExtractor, EntityLinker
        
        # Test triplet extraction (requires spacy model)
        try:
            extractor = TripletExtractor()
            triplets = extractor.extract("Paris is the capital of France")
            print(f"✅ Triplet extraction works: {triplets}")
        except Exception as e:
            print(f"⚠️  Triplet extraction needs spacy model: {e}")
        
        # Test entity linker
        linker = EntityLinker()
        print(f"✅ Entity linker initialized: {linker}")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing factcheck package installation")
    print("=" * 60)
    print()
    
    print("1. Testing imports...")
    imports_ok = test_imports()
    print()
    
    if imports_ok:
        print("2. Testing basic functionality...")
        test_basic_functionality()
        print()
    
    print("=" * 60)
    if imports_ok:
        print("✅ Package installation verified!")
        print()
        print("Next steps:")
        print("  1. Download spacy model: python -m spacy download en_core_web_sm")
        print("  2. Download your trained model from MLflow/DagsHub")
        print("  3. Use in your API (see API_INTEGRATION.md)")
    else:
        print("❌ Package not installed correctly")
        print()
        print("To install:")
        print("  pip install git+https://github.com/MarcoSrhl/factcheck.git")
    print("=" * 60)
