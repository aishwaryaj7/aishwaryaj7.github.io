#!/usr/bin/env python3
"""
Simplified test script for dataset management functionality.
Tests dataset initialization and sample preparation.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Simple mock classes to avoid complex dependencies
class MockSettings:
    class DataDirectories:
        data_dir = "./data"
    
    data_directories = DataDirectories()

# Import our dataset manager
from rag_chatbot.data.dataset_manager import DatasetManager


async def test_dataset_initialization():
    """Test dataset initialization and analysis."""
    print("🔍 Testing dataset initialization...")
    
    settings = MockSettings()
    dataset_manager = DatasetManager(settings)
    
    # Initialize datasets
    datasets_info = await dataset_manager.initialize_datasets()
    
    print("\n" + "="*60)
    print("📊 DATASET ANALYSIS RESULTS")
    print("="*60)
    
    # Print invoice dataset info
    invoice_info = datasets_info.get("invoices", {})
    print(f"\n📄 INVOICE DATASET:")
    print(f"   Available: {invoice_info.get('available', False)}")
    if invoice_info.get('available'):
        print(f"   Total Files: {invoice_info.get('total_files', 0)}")
        print(f"   Layouts: {len(invoice_info.get('layouts', []))}")
        for layout in invoice_info.get('layouts', []):
            print(f"     - {layout['name']}: {layout['file_count']} files")
    else:
        print(f"   Reason: {invoice_info.get('reason', 'unknown')}")
    
    # Print contract dataset info
    contract_info = datasets_info.get("contracts", {})
    print(f"\n📋 CONTRACT DATASET:")
    print(f"   Available: {contract_info.get('available', False)}")
    if contract_info.get('available'):
        print(f"   Files: {contract_info.get('files', [])}")
        print(f"   Description: {contract_info.get('description', 'N/A')}")
    else:
        print(f"   Reason: {contract_info.get('reason', 'unknown')}")
    
    print(f"\n📈 TOTAL DOCUMENTS: {datasets_info.get('total_documents', 0)}")
    
    return dataset_manager, datasets_info


async def test_sample_preparation(dataset_manager: DatasetManager):
    """Test sample data preparation."""
    print("\n🔬 Testing sample data preparation...")
    
    try:
        # Prepare sample data
        sample_files = await dataset_manager.prepare_sample_data(
            num_invoices=3, 
            num_contracts=2
        )
        
        print("\n" + "="*60)
        print("🧪 SAMPLE DATA PREPARATION")
        print("="*60)
        
        print(f"\n📄 Invoice samples: {len(sample_files['invoices'])}")
        for file_path in sample_files['invoices']:
            print(f"   - {Path(file_path).name}")
        
        print(f"\n📋 Contract samples: {len(sample_files['contracts'])}")
        for file_path in sample_files['contracts']:
            print(f"   - {Path(file_path).name}")
        
        # Validate samples
        all_samples = sample_files['invoices'] + sample_files['contracts']
        if all_samples:
            validation_results = await dataset_manager.validate_sample_data(all_samples)
            
            print(f"\n✅ Valid samples: {len(validation_results['valid_files'])}")
            print(f"❌ Invalid samples: {len(validation_results['invalid_files'])}")
            
            if validation_results['errors']:
                print("\n🚨 Validation errors:")
                for error in validation_results['errors']:
                    print(f"   - {error}")
        else:
            print("⚠️ No samples could be prepared - datasets may not be available")
        
        return sample_files
        
    except Exception as e:
        print(f"❌ Sample preparation failed: {e}")
        return {"invoices": [], "contracts": []}


async def inspect_sample_data(sample_files: dict):
    """Inspect the prepared sample data."""
    print("\n🔍 Inspecting prepared sample data...")
    
    print("\n" + "="*60)
    print("📄 SAMPLE DATA INSPECTION")
    print("="*60)
    
    # Inspect invoice samples
    for file_path in sample_files['invoices'][:2]:  # First 2 invoice samples
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"\n📄 {Path(file_path).name}")
            print(f"   Type: {data.get('type', 'unknown')}")
            print(f"   Layout: {data.get('layout', 'unknown')}")
            print(f"   Entities: {data.get('metadata', {}).get('num_entities', 0)}")
            print(f"   Content length: {len(data.get('text_content', ''))} chars")
            
            # Show preview of OCR data
            ocr_data = data.get('ocr_data', [])
            if ocr_data:
                print(f"   OCR preview (first 5 entities):")
                for i, entity in enumerate(ocr_data[:5]):
                    text = entity.get('Text', 'N/A')
                    tag = entity.get('Tag', 'N/A')
                    print(f"     {i+1}. '{text}' -> {tag}")
            
        except Exception as e:
            print(f"   ❌ Error inspecting {file_path}: {e}")
    
    # Inspect contract samples
    for file_path in sample_files['contracts'][:1]:  # First contract sample
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"\n📋 {Path(file_path).name}")
            print(f"   Type: {data.get('type', 'unknown')}")
            print(f"   Title: {data.get('title', 'unknown')}")
            print(f"   Q&A pairs: {len(data.get('qa_pairs', []))}")
            print(f"   Content length: {len(data.get('text_content', ''))} chars")
            
            # Show preview of Q&A pairs
            qa_pairs = data.get('qa_pairs', [])
            if qa_pairs:
                print(f"   Q&A preview (first 3 pairs):")
                for i, qa in enumerate(qa_pairs[:3]):
                    question = qa.get('question', 'N/A')
                    print(f"     {i+1}. Q: {question[:80]}...")
                    answers = qa.get('answers', [])
                    if answers:
                        answer_text = answers[0].get('text', 'N/A')
                        print(f"        A: {answer_text[:60]}...")
            
            # Show content preview
            content = data.get('text_content', '')
            if content:
                print(f"   Content preview: {content[:200]}...")
            
        except Exception as e:
            print(f"   ❌ Error inspecting {file_path}: {e}")


async def main():
    """Main test function."""
    print("🚀 STARTING DATASET MANAGEMENT TEST")
    print("="*60)
    
    try:
        # Test 1: Dataset initialization
        dataset_manager, datasets_info = await test_dataset_initialization()
        
        # Test 2: Sample preparation
        sample_files = await test_sample_preparation(dataset_manager)
        
        # Test 3: Sample data inspection
        if any(sample_files.values()):
            await inspect_sample_data(sample_files)
        
        # Final statistics
        stats = await dataset_manager.get_dataset_statistics()
        
        print("\n" + "="*60)
        print("📈 FINAL STATISTICS")
        print("="*60)
        
        available_datasets = sum(1 for d in [datasets_info['invoices'], datasets_info['contracts']] 
                               if d.get('available'))
        print(f"📊 Datasets available: {available_datasets}/2")
        print(f"🧪 Invoice samples: {stats['samples_prepared']['invoices']}")
        print(f"🧪 Contract samples: {stats['samples_prepared']['contracts']}")
        print(f"💾 Data directory size: {stats['storage_info']['data_dir_size']}")
        print(f"💾 Samples directory size: {stats['storage_info']['samples_dir_size']}")
        
        if available_datasets > 0:
            print(f"\n✅ Dataset test completed successfully!")
            print(f"   - {available_datasets} dataset(s) available")
            print(f"   - {sum(len(files) for files in sample_files.values())} sample(s) prepared")
        else:
            print(f"\n⚠️ Dataset test completed with warnings!")
            print(f"   - No datasets found - check if data was downloaded correctly")
        
    except Exception as e:
        print(f"\n❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 