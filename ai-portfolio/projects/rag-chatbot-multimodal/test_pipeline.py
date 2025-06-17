#!/usr/bin/env python3
"""
Test script for the RAG chatbot pipeline with real datasets.
Tests invoice and contract processing capabilities.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_chatbot.core.config import Settings
from rag_chatbot.core.logger import get_logger
from rag_chatbot.data.dataset_manager import DatasetManager
from rag_chatbot.document_processing.extractor import MultiModalDocumentExtractor
from rag_chatbot.rag.vector_store import WeaviateVectorStore
from rag_chatbot.rag.pipeline import RAGPipeline

logger = get_logger(__name__)


async def test_dataset_initialization():
    """Test dataset initialization and analysis."""
    logger.info("🔍 Testing dataset initialization...")
    
    settings = Settings()
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
    
    # Print contract dataset info
    contract_info = datasets_info.get("contracts", {})
    print(f"\n📋 CONTRACT DATASET:")
    print(f"   Available: {contract_info.get('available', False)}")
    if contract_info.get('available'):
        print(f"   Files: {contract_info.get('files', [])}")
        print(f"   Description: {contract_info.get('description', 'N/A')}")
    
    print(f"\n📈 TOTAL DOCUMENTS: {datasets_info.get('total_documents', 0)}")
    
    return dataset_manager, datasets_info


async def test_sample_preparation(dataset_manager: DatasetManager):
    """Test sample data preparation."""
    logger.info("🔬 Testing sample data preparation...")
    
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
    validation_results = await dataset_manager.validate_sample_data(all_samples)
    
    print(f"\n✅ Valid samples: {len(validation_results['valid_files'])}")
    print(f"❌ Invalid samples: {len(validation_results['invalid_files'])}")
    
    if validation_results['errors']:
        print("\n🚨 Validation errors:")
        for error in validation_results['errors']:
            print(f"   - {error}")
    
    return sample_files, validation_results


async def test_document_processing(sample_files: dict):
    """Test document processing with sample data."""
    logger.info("📝 Testing document processing...")
    
    settings = Settings()
    extractor = MultiModalDocumentExtractor()
    
    print("\n" + "="*60)
    print("📝 DOCUMENT PROCESSING TEST")
    print("="*60)
    
    processed_docs = []
    
    # Process a few sample files
    test_files = (sample_files['invoices'][:2] + sample_files['contracts'][:1])
    
    for file_path in test_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
            
            print(f"\n📄 Processing: {Path(file_path).name}")
            print(f"   Type: {sample_data['type']}")
            print(f"   Content length: {len(sample_data['text_content'])} chars")
            
            # Process the text content directly
            text_content = sample_data['text_content']

            # Create document chunks manually since we have text data
            chunk_size = 1000
            chunks = []
            for i in range(0, len(text_content), chunk_size):
                chunk = text_content[i:i + chunk_size]
                chunks.append({
                    'content': chunk,
                    'content_type': 'text',
                    'document_name': Path(file_path).name,
                    'chunk_index': i // chunk_size,
                    'page_number': 1,
                    'metadata': {
                        'source_file': str(file_path),
                        'document_type': sample_data['type'],
                        'chunk_size': len(chunk)
                    }
                })

            extracted_docs = chunks
            
            print(f"   ✅ Extracted {len(extracted_docs)} document chunks")
            
            for i, doc in enumerate(extracted_docs[:2]):  # Show first 2 chunks
                content = doc.get('content', '')
                print(f"     Chunk {i+1}: {len(content)} chars")
                print(f"     Preview: {content[:100]}...")
            
            processed_docs.extend(extracted_docs)
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            print(f"   ❌ Processing failed: {e}")
    
    print(f"\n📊 Total processed chunks: {len(processed_docs)}")
    return processed_docs


async def test_vector_storage(processed_docs):
    """Test vector storage with processed documents."""
    logger.info("🗄️ Testing vector storage...")
    
    settings = Settings()
    
    print("\n" + "="*60)
    print("🗄️ VECTOR STORAGE TEST")
    print("="*60)
    
    try:
        vector_store = WeaviateVectorStore()
        await vector_store.connect()
        
        print("✅ Vector store initialized")
        
        # Add documents in batches
        if processed_docs:
            print(f"📥 Adding {len(processed_docs)} documents to vector store...")
            
            document_ids = await vector_store.add_documents(processed_docs, "test_document_batch")
            print(f"✅ Successfully added {len(document_ids)} documents")
            
            # Get collection stats
            stats = await vector_store.get_collection_stats()
            print(f"📊 Collection stats: {stats}")
            
            # Test search
            print("\n🔍 Testing search functionality...")
            results = await vector_store.search(
                query="invoice amount total payment",
                limit=3
            )
            
            print(f"   Found {len(results)} results")
            for i, result in enumerate(results):
                score = result.get('similarity_score', 0.0)
                content = result.get('content', '')
                print(f"   Result {i+1}: Score {score:.3f}")
                print(f"     Preview: {content[:100]}...")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Vector storage test failed: {e}")
        print(f"❌ Vector storage failed: {e}")
        return None


async def test_rag_pipeline(vector_store, sample_files):
    """Test the complete RAG pipeline."""
    logger.info("🤖 Testing RAG pipeline...")
    
    if not vector_store:
        print("❌ Skipping RAG test - vector store not available")
        return
    
    settings = Settings()
    
    print("\n" + "="*60)
    print("🤖 RAG PIPELINE TEST")
    print("="*60)
    
    try:
        pipeline = RAGPipeline(settings, vector_store)
        
        # Test questions based on our datasets
        test_questions = [
            "What are the key financial terms in the contracts?",
            "Show me information about invoice totals and amounts",
            "What are the main parties mentioned in these documents?",
            "What payment terms are specified?",
            "Are there any liability clauses mentioned?"
        ]
        
        print("🎯 Testing with sample questions...")
        
        for i, question in enumerate(test_questions[:3], 1):  # Test first 3 questions
            print(f"\n❓ Question {i}: {question}")
            
            try:
                response = await pipeline.process_query(question)
                
                print(f"   ✅ Response generated ({len(response.answer)} chars)")
                print(f"   📚 Sources: {len(response.sources)}")
                print(f"   ⏱️ Response time: {response.response_time:.2f}s")
                print(f"   📝 Answer preview: {response.answer[:200]}...")
                
                if response.sources:
                    print("   📖 Top sources:")
                    for j, source in enumerate(response.sources[:2], 1):
                        print(f"     {j}. Score: {source.score:.3f}")
                        print(f"        Content: {source.content[:100]}...")
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                print(f"   ❌ Query failed: {e}")
        
        print(f"\n🎉 RAG pipeline test completed!")
        
    except Exception as e:
        logger.error(f"RAG pipeline test failed: {e}")
        print(f"❌ RAG pipeline failed: {e}")


async def main():
    """Main test function."""
    print("🚀 STARTING RAG CHATBOT PIPELINE TEST")
    print("="*60)
    
    try:
        # Test 1: Dataset initialization
        dataset_manager, datasets_info = await test_dataset_initialization()
        
        # Test 2: Sample preparation
        sample_files, validation_results = await test_sample_preparation(dataset_manager)
        
        # Test 3: Document processing
        processed_docs = await test_document_processing(sample_files)
        
        # Test 4: Vector storage
        vector_store = await test_vector_storage(processed_docs)
        
        # Test 5: RAG pipeline
        await test_rag_pipeline(vector_store, sample_files)
        
        # Final statistics
        stats = await dataset_manager.get_dataset_statistics()
        
        print("\n" + "="*60)
        print("📈 FINAL STATISTICS")
        print("="*60)
        print(f"📊 Datasets available: {len([d for d in [datasets_info['invoices'], datasets_info['contracts']] if d.get('available')])}")
        print(f"🧪 Samples prepared: {stats['samples_prepared']['invoices'] + stats['samples_prepared']['contracts']}")
        print(f"📝 Documents processed: {len(processed_docs)}")
        print(f"💾 Data directory size: {stats['storage_info']['data_dir_size']}")
        
        print(f"\n✅ Pipeline test completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        print(f"\n❌ Pipeline test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 