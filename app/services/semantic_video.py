#!/usr/bin/env python3
"""
Semantic video selection service for intelligent video-text matching
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from loguru import logger
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Global model instance
_model = None
_model_name = None
_model_load_fails = 0
_max_model_retries = 3

try:
    from app.services import image_similarity
    IMAGE_SIMILARITY_AVAILABLE = True
except ImportError:
    IMAGE_SIMILARITY_AVAILABLE = False
    logger.warning("Image similarity service not available - install transformers, torch, and pillow for image similarity features")

def load_model(model_name: str = "all-mpnet-base-v2"):
    """Load the semantic search model"""
    global _model, _model_name, _model_load_fails
    
    # Check if we've had too many failures
    if _model_load_fails >= _max_model_retries:
        logger.error(f"❌ Maximum model loading retries ({_max_model_retries}) exceeded for semantic model")
        raise Exception(f"Semantic model loading failed {_model_load_fails} times, giving up")
    
    if _model is None or _model_name != model_name:
        try:
            logger.info(f"🤖 Loading semantic search model: {model_name}")
            logger.info("📦 This may take a moment on first run (downloading model)...")
            
            # Force CPU usage to avoid GPU hanging issues
            logger.info("🖥️  Forcing CPU-only mode for SentenceTransformer to avoid GPU issues")
            _model = SentenceTransformer(model_name, device='cpu')
            _model_name = model_name
            
            # Reset failure count on successful load
            _model_load_fails = 0
            
            logger.success(f"✅ Semantic search model loaded successfully: {model_name} (CPU-only)")
            logger.info(f"🔧 Model max sequence length: {_model.max_seq_length}")
            
        except Exception as e:
            _model_load_fails += 1
            logger.error(f"Failed to load semantic model {model_name} (attempt {_model_load_fails}/{_max_model_retries}): {e}")
            
            # Try resetting if we have failures
            if _model_load_fails < _max_model_retries:
                reset_semantic_model()
            
            raise
    
    return _model

def segment_script_into_sentences(script: str, min_length: int = 25, max_length: int = 150) -> List[str]:
    """Segment script into sentences with minimum and maximum length"""
    logger.info(f"📝 Segmenting script using method: sentences")
    logger.info(f"📏 Minimum segment length: {min_length} characters")
    logger.info(f"📏 Maximum segment length: {max_length} characters")
    logger.info(f"📄 Original script length: {len(script)} characters")
    logger.debug(f"📄 Original script: '{script[:100]}...'")
    
    # Split by sentence endings first
    sentences = re.split(r'[.!?]+', script)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    logger.debug("🔍 Found {} initial sentences:".format(len(sentences)))
    for i, sentence in enumerate(sentences, 1):
        logger.debug(f"   {i}. '{sentence[:100]}...' ({len(sentence)} chars)")
    
    # Process each sentence - split long ones by commas if needed
    processed_sentences = []
    
    for sentence in sentences:
        if len(sentence) <= max_length:
            processed_sentences.append(sentence)
        else:
            # Long sentence - split by commas and merge to appropriate lengths
            logger.debug(f"📐 Long sentence detected ({len(sentence)} chars), splitting by commas...")
            comma_parts = [part.strip() for part in sentence.split(',') if part.strip()]
            
            # Merge comma parts to create segments of appropriate length
            current_segment = ""
            for part in comma_parts:
                if current_segment:
                    test_segment = current_segment + ", " + part
                else:
                    test_segment = part
                
                if len(test_segment) <= max_length:
                    current_segment = test_segment
                else:
                    # Current segment is good length, save it and start new one
                    if current_segment:
                        processed_sentences.append(current_segment)
                        current_segment = part
                    else:
                        # Even single part is too long, save it anyway
                        processed_sentences.append(part)
                        current_segment = ""
            
            # Add remaining segment
            if current_segment:
                processed_sentences.append(current_segment)
    
    # Now merge short sentences as before
    merged_sentences = []
    current_sentence = ""
    
    for sentence in processed_sentences:
        if current_sentence:
            test_sentence = current_sentence + ". " + sentence
        else:
            test_sentence = sentence
            
        if len(test_sentence) >= min_length and len(test_sentence) <= max_length:
            if current_sentence:
                merged_sentences.append(current_sentence)
                current_sentence = sentence
            else:
                merged_sentences.append(sentence)
                current_sentence = ""
        elif len(test_sentence) > max_length:
            # Would be too long, save current and start with this sentence
            if current_sentence:
                merged_sentences.append(current_sentence)
            current_sentence = sentence
        else:
            # Too short, keep building
            current_sentence = test_sentence
    
    # Add remaining sentence
    if current_sentence:
        merged_sentences.append(current_sentence)
    
    logger.info(f"✅ Final segmentation: {len(merged_sentences)} segments after processing")
    for i, segment in enumerate(merged_sentences, 1):
        logger.info(f"   📝 Segment {i}: '{segment[:60]}...' ({len(segment)} chars)")
    
    return merged_sentences

def calculate_similarity(sentence: str, video_text: str) -> float:
    """Calculate semantic similarity between sentence and video text"""
    try:
        model = load_model()
        
        # Reduced logging - only log device info once per session
        if not hasattr(calculate_similarity, '_device_logged'):
            try:
                if hasattr(model, 'device'):
                    device_info = str(model.device)
                elif hasattr(model, '_modules'):
                    first_module = next(iter(model._modules.values()))
                    if hasattr(first_module, 'device'):
                        device_info = str(first_module.device)
                    else:
                        device_info = "unknown"
                else:
                    device_info = "unknown"
                logger.info(f"🖥️  Text similarity model device: {device_info}")
                calculate_similarity._device_logged = True
            except Exception as device_error:
                logger.debug(f"⚠️  Could not determine model device: {device_error}")
        
        # Ensure model is on CPU
        if hasattr(model, 'to'):
            model = model.to('cpu')
        
        # Encode both texts - minimal logging
        try:
            sentence_embedding = model.encode([sentence], device='cpu')
            video_embedding = model.encode([video_text], device='cpu')
        except Exception as encode_error:
            logger.warning(f"⚠️ Encoding error, trying without explicit device: {encode_error}")
            sentence_embedding = model.encode([sentence])
            video_embedding = model.encode([video_text])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(sentence_embedding, video_embedding)[0][0]
        
        return float(similarity)
        
    except Exception as e:
        logger.error(f"❌ Error calculating text similarity: {e}")
        import traceback
        logger.error(f"❌ Text similarity traceback: {traceback.format_exc()}")
        return 0.1

def find_best_video_for_sentence(
    sentence: str, 
    video_metadata: List[Dict], 
    used_videos: Dict[str, int],
    similarity_threshold: float = 0.5,
    diversity_threshold: int = 5,
    max_video_reuse: int = 2,
    enable_image_similarity: bool = False,
    image_similarity_threshold: float = 0.7,
    image_similarity_model: str = "clip-vit-base-patch32"
) -> Optional[Dict]:
    """Find the best video for a given sentence with strong diversity controls"""
    logger.info(f"🔍 Finding best video for sentence: '{sentence[:60]}...'")
    logger.info(f"📊 Analyzing {len(video_metadata)} available videos")
    
    # Calculate all similarities and scores once
    video_scores = []
    
    for i, video_meta in enumerate(video_metadata, 1):
        try:
            video_path = video_meta['video_path']
            search_term = video_meta['search_term']
            
            # Minimal logging - only every 5th video
            if i % 5 == 1 or i == len(video_metadata):
                logger.debug(f"🔄 Processing video {i}/{len(video_metadata)}: {os.path.basename(video_path)}")
            
            # Calculate similarity once - no debug logging
            similarity = calculate_similarity(sentence, search_term)
            
            # Calculate image similarity if enabled and available
            image_similarity_score = 0.0
            if enable_image_similarity and IMAGE_SIMILARITY_AVAILABLE:
                try:
                    # Check if image similarity model is healthy before proceeding
                    if hasattr(image_similarity, 'is_model_healthy') and not image_similarity.is_model_healthy():
                        logger.warning(f"⚠️ Image similarity model unhealthy, skipping for video {i}")
                        image_similarity_score = 0.0
                    else:
                        image_similarity_score = image_similarity.calculate_video_image_similarity(
                            sentence, video_meta, image_similarity_model
                        )
                        
                except Exception as e:
                    logger.error(f"❌ Failed to calculate image similarity for video {i}: {e}")
                    import traceback
                    logger.error(f"❌ Image similarity traceback: {traceback.format_exc()}")
                    
                    # Try to reset the model if we have repeated failures
                    try:
                        if hasattr(image_similarity, 'force_model_reset') and "timeout" in str(e).lower():
                            logger.warning("🔄 Timeout detected, forcing model reset...")
                            image_similarity.force_model_reset()
                    except:
                        pass
                        
                    image_similarity_score = 0.0
            
            # Combine text and image similarity scores
            if enable_image_similarity and IMAGE_SIMILARITY_AVAILABLE:
                # Weight: 30% text similarity, 70% image similarity
                combined_similarity = (0.3 * similarity) + (0.7 * image_similarity_score)
            else:
                combined_similarity = similarity
            
            # Enhanced diversity penalty system based on max_video_reuse
            usage_count = used_videos.get(video_path, 0)
            
            # Dynamic penalty based on max_video_reuse setting
            if usage_count == 0:
                diversity_penalty = 0.0  # No penalty for first use
            elif usage_count == 1 and max_video_reuse >= 2:
                diversity_penalty = 0.2  # Light penalty for second use
            elif usage_count == 2 and max_video_reuse >= 3:
                diversity_penalty = 0.4  # Moderate penalty for third use
            elif usage_count == 3 and max_video_reuse >= 4:
                diversity_penalty = 0.6  # Heavy penalty for fourth use
            else:
                diversity_penalty = 1.0  # Eliminate from consideration
            
            # Additional penalty if we've exceeded max_video_reuse
            if usage_count >= max_video_reuse:
                diversity_penalty = 1.0  # Completely eliminate from consideration
            
            final_score = combined_similarity - diversity_penalty
            
            # Store all results for later use
            video_scores.append({
                'video': video_meta,
                'text_similarity': similarity,
                'image_similarity': image_similarity_score,
                'combined_similarity': combined_similarity,
                'usage': usage_count,
                'penalty': diversity_penalty,
                'final_score': final_score
            })
            
        except Exception as video_error:
            logger.error(f"❌ Critical error processing video {i}/{len(video_metadata)}: {video_error}")
            import traceback
            logger.error(f"❌ Video processing traceback: {traceback.format_exc()}")
            
            # Add a minimal entry to keep processing
            video_scores.append({
                'video': video_meta,
                'text_similarity': 0.0,
                'image_similarity': 0.0,
                'combined_similarity': 0.0,
                'usage': used_videos.get(video_meta['video_path'], 0),
                'penalty': 1.0,  # High penalty for failed videos
                'final_score': -1.0  # Very low score
            })
            
            logger.warning(f"⚠️ Added fallback entry for failed video {i}, continuing...")
            continue
    
    # Find best video from calculated scores
    best_video = None
    best_score = -1
    
    for score_data in video_scores:
        if score_data['final_score'] > best_score and (score_data['final_score'] > 0 or best_score < 0):
            best_score = score_data['final_score']
            best_video = score_data['video']
    
    # Log metadata summary
    videos_with_metadata = len([v for v in video_metadata if v.get('search_term')])
    logger.info("📈 Video metadata summary:")
    logger.info(f"   ✅ Videos with metadata: {videos_with_metadata}/{len(video_metadata)} ({videos_with_metadata/len(video_metadata)*100:.1f}%)")
    
    # Log usage statistics
    if used_videos:
        usage_stats = {}
        for path, count in used_videos.items():
            usage_stats[count] = usage_stats.get(count, 0) + 1
        logger.info("🔄 Video usage statistics:")
        for usage_count, video_count in sorted(usage_stats.items()):
            logger.info(f"   Used {usage_count} times: {video_count} videos")
    
    # Sort scores for logging top candidates  
    video_scores.sort(key=lambda x: x['final_score'], reverse=True)
    
    logger.info("🏆 Top video candidates:")
    for i, score_data in enumerate(video_scores[:3], 1):
        video = score_data['video']
        text_sim = score_data['text_similarity']
        image_sim = score_data['image_similarity']
        combined_sim = score_data['combined_similarity']
        usage = score_data['usage']
        penalty = score_data['penalty']
        score = score_data['final_score']
        logger.info(f"   {i}. {os.path.basename(video['video_path'])}: text_similarity={text_sim:.3f}, image_similarity={image_sim:.3f}, combined_similarity={combined_sim:.3f}, used={usage}/{max_video_reuse}x, final_score={score:.3f}")
    
    if best_score < similarity_threshold:
        logger.warning(f"⚠️  Best similarity ({best_score:.3f}) below threshold ({similarity_threshold}), using anyway")
    
    if best_video:
        usage = used_videos.get(best_video['video_path'], 0)
        
        # Find the detailed scores for the selected video
        selected_video_scores = None
        for score_data in video_scores:
            if score_data['video']['video_path'] == best_video['video_path']:
                selected_video_scores = score_data
                break
        
        logger.success(f"🎯 SELECTED: {os.path.basename(best_video['video_path'])} with similarity {best_score:.3f} (will be used {usage + 1}/{max_video_reuse} times)")
    else:
        logger.error("❌ No suitable video found - all videos may be overused")
        selected_video_scores = None
    
    logger.info("=" * 80)
    
    # Return both the video and its detailed scores
    return best_video, selected_video_scores

def select_videos_for_script(
    script: str,
    video_metadata: List[Dict],
    audio_duration: float,
    similarity_threshold: float = 0.5,
    diversity_threshold: int = 5,
    max_video_reuse: int = 2,
    min_segment_length: int = 25,
    semantic_model: str = "all-mpnet-base-v2",
    enable_image_similarity: bool = False,
    image_similarity_threshold: float = 0.7,
    image_similarity_model: str = "clip-vit-base-patch32"
) -> List[Dict]:
    """Select videos for script segments using semantic matching"""
    
    logger.info("🎬" + "=" * 50 + " SEMANTIC VIDEO SELECTION " + "=" * 50)
    logger.info("🎯 Starting semantic video selection for script")
    logger.info(f"📺 Available video pool: {len(video_metadata)} videos")
    logger.info(f"⏱️  Target audio duration: {audio_duration:.2f} seconds")
    
    # Load the specified semantic model
    load_model(semantic_model)
    
    # Configuration logging
    logger.info("⚙️  Configuration:")
    logger.info(f"   🎯 Similarity threshold: {similarity_threshold}")
    logger.info(f"   🔄 Diversity threshold: {diversity_threshold}")
    logger.info(f"   🔁 Max video reuse: {max_video_reuse}")
    logger.info(f"   🤖 Semantic model: {semantic_model}")
    
    # Image similarity configuration
    if enable_image_similarity:
        if IMAGE_SIMILARITY_AVAILABLE:
            logger.info(f"   🖼️  Image similarity: ENABLED")
            logger.info(f"   🎨 Image similarity threshold: {image_similarity_threshold}")
            logger.info(f"   🤖 Image similarity model: {image_similarity_model}")
        else:
            logger.warning("   🖼️  Image similarity: REQUESTED but NOT AVAILABLE (missing dependencies)")
            logger.warning("   📦 Install: pip install transformers torch pillow")
    else:
        logger.info(f"   🖼️  Image similarity: DISABLED")
    
    # Segment script
    # Use max_length of 120 to create more segments from long sentences
    segments = segment_script_into_sentences(script, min_segment_length, max_length=120)
    logger.info("=" * 100)
    
    # Calculate duration per segment
    duration_per_segment = audio_duration / len(segments)
    logger.info(f"⏱️  Approximate duration per segment: {duration_per_segment:.2f} seconds")
    logger.info("=" * 100)
    
    selected_videos = []
    used_videos = {}
    
    for i, segment in enumerate(segments, 1):
        logger.info(f"🔄 PROCESSING SEGMENT {i}/{len(segments)}")
        
        best_video, selected_video_scores = find_best_video_for_sentence(
            segment, 
            video_metadata, 
            used_videos,
            similarity_threshold,
            diversity_threshold,
            max_video_reuse,
            enable_image_similarity,
            image_similarity_threshold,
            image_similarity_model
        )
        
        if best_video:
            selected_videos.append({
                'video_path': best_video['video_path'],
                'segment': segment,
                'search_term': best_video['search_term'],
                'duration': duration_per_segment
            })
            
            # Update usage count
            video_path = best_video['video_path']
            used_videos[video_path] = used_videos.get(video_path, 0) + 1
            
            # Enhanced logging with both similarity scores
            if selected_video_scores and enable_image_similarity and IMAGE_SIMILARITY_AVAILABLE:
                logger.success(f"✅ SEGMENT {i} COMPLETED: Selected {os.path.basename(best_video['video_path'])} (text: {selected_video_scores['text_similarity']:.3f}, image: {selected_video_scores['image_similarity']:.3f}, combined: {selected_video_scores['combined_similarity']:.3f})")
            else:
                logger.success(f"✅ SEGMENT {i} COMPLETED: Selected {os.path.basename(best_video['video_path'])} (text similarity: {selected_video_scores['text_similarity']:.3f})")
        else:
            logger.error(f"❌ SEGMENT {i} FAILED: No suitable video found")
    
    # Final diversity report
    logger.info("🎬" + "=" * 100)
    logger.success(f"🎉 SEMANTIC SELECTION COMPLETED: {len(selected_videos)}/{len(segments)} segments matched")
    
    # Log final usage statistics
    if used_videos:
        logger.info("📊 Final video usage distribution:")
        usage_distribution = {}
        for path, count in used_videos.items():
            usage_distribution[count] = usage_distribution.get(count, 0) + 1
        
        for usage_count, video_count in sorted(usage_distribution.items()):
            percentage = (video_count / len(used_videos)) * 100
            logger.info(f"   Used {usage_count} times: {video_count} videos ({percentage:.1f}%)")
        
        # Check if diversity goals were met
        unique_videos_used = len(used_videos)
        total_segments = len(segments)
        diversity_ratio = unique_videos_used / total_segments if total_segments > 0 else 0
        
        logger.info(f"🎯 Diversity metrics:")
        logger.info(f"   📹 Unique videos used: {unique_videos_used}")
        logger.info(f"   📝 Total segments: {total_segments}")
        logger.info(f"   📊 Diversity ratio: {diversity_ratio:.2f} ({diversity_ratio*100:.1f}%)")
        
        if diversity_ratio >= 0.8:
            logger.success("✅ Excellent diversity achieved!")
        elif diversity_ratio >= 0.6:
            logger.info("✅ Good diversity achieved")
        elif diversity_ratio >= 0.4:
            logger.warning("⚠️  Moderate diversity - consider increasing video pool")
        else:
            logger.warning("⚠️  Low diversity - recommend more diverse search terms or larger video pool")
    
    return selected_videos

def get_metadata_path(video_path: str) -> str:
    """Get metadata file path for a video"""
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f"{video_name}_metadata.json")

def save_video_metadata(video_path: str, search_term: str, additional_info: Dict = None):
    """Save metadata for a video file"""
    metadata = {
        'video_path': video_path,
        'search_term': search_term,
        'file_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0,
        'created_at': os.path.getctime(video_path) if os.path.exists(video_path) else 0
    }
    
    if additional_info:
        metadata.update(additional_info)
    
    metadata_path = get_metadata_path(video_path)
    
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved metadata for {video_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata for {video_path}: {e}")

def load_video_metadata(video_path: str) -> Optional[Dict]:
    """Load metadata for a video file"""
    metadata_path = get_metadata_path(video_path)
    
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.debug(f"Loaded metadata for {video_path}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to load metadata for {video_path}: {e}")
        return None

def get_video_metadata_list(video_paths: List[str]) -> List[Dict]:
    """Get metadata for a list of video files"""
    metadata_list = []
    
    for video_path in video_paths:
        metadata = load_video_metadata(video_path)
        if metadata:
            metadata_list.append(metadata)
        else:
            # Create default metadata if none exists
            logger.warning(f"No metadata found for {video_path}, using filename as search term")
            filename = os.path.splitext(os.path.basename(video_path))[0]
            metadata = {
                'video_path': video_path,
                'search_term': filename,
                'file_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0,
                'created_at': os.path.getctime(video_path) if os.path.exists(video_path) else 0
            }
            metadata_list.append(metadata)
    
    return metadata_list

def reset_semantic_model():
    """Reset the semantic model if it gets into a bad state"""
    global _model, _model_name, _model_load_fails
    logger.warning("🔄 Resetting semantic model due to errors")
    
    try:
        if _model is not None:
            del _model
        _model = None
        _model_name = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("✅ Semantic model reset completed")
        
    except Exception as e:
        logger.error(f"❌ Error during semantic model reset: {e}")
        _model_load_fails += 1

def is_semantic_model_healthy() -> bool:
    """Check if the semantic model is in a healthy state"""
    global _model_load_fails
    return _model_load_fails < _max_model_retries 