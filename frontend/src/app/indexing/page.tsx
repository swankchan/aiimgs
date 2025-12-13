'use client';

import { useState } from 'react';
import { apiClient } from '@/lib/api';
import { FolderOpen, Loader2, CheckCircle, AlertCircle, Upload, FileText } from 'lucide-react';

export default function IndexingPage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState('');
  
  // Image upload states
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [uploadingImages, setUploadingImages] = useState(false);
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [uploadError, setUploadError] = useState('');
  const [imageCaption, setImageCaption] = useState('');
  const [imageKeywords, setImageKeywords] = useState('');
  const [lastUploadedCount, setLastUploadedCount] = useState<number>(0);
  
  // PDF processing states
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [pdfProcessing, setPdfProcessing] = useState(false);
  const [pdfResult, setPdfResult] = useState<any>(null);
  const [pdfError, setPdfError] = useState('');
  const [useAI, setUseAI] = useState(true);
  const [pdfCaption, setPdfCaption] = useState('');
  const [pdfKeywords, setPdfKeywords] = useState('');
  
  // Review/Edit state for PDF extracted images
  const [showReviewModal, setShowReviewModal] = useState(false);
  const [reviewData, setReviewData] = useState<any>(null);
  const [editedCaption, setEditedCaption] = useState('');
  const [editedKeywords, setEditedKeywords] = useState<string[]>([]);
  const [keywordsByMethod, setKeywordsByMethod] = useState<{[key: string]: string[]}>({});

  const handleIndex = async () => {
    setLoading(true);
    setError('');
    setResult(null);

    try {
      // Index the server's image folder (hardcoded path)
      const response = await apiClient.indexFolder('C:\\AIIMGS\\images', 'clip-vit-b-16');
      setResult(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Indexing failed');
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = async () => {
    if (imageFiles.length === 0) {
      setUploadError('Please select at least one image');
      return;
    }

    setUploadingImages(true);
    setUploadError('');
    setUploadResult(null);

    try {
      // Capture count before clearing state for display
      setLastUploadedCount(imageFiles.length);
      const response = await apiClient.uploadImages(imageFiles, imageCaption, imageKeywords);
      setUploadResult(response);
      
      // Clear selected files
      setImageFiles([]);
      setImageCaption('');
      setImageKeywords('');
      
      // Automatically trigger indexing after upload
      setTimeout(() => {
        handleIndex();
      }, 500);
    } catch (err: any) {
      setUploadError(err.response?.data?.detail || 'Image upload failed');
    } finally {
      setUploadingImages(false);
    }
  };

  const handleImageFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      setImageFiles(files);
      setUploadError('');
      setUploadResult(null); // Clear previous upload result when selecting new files
    }
  };

  const removeImage = (index: number) => {
    setImageFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handlePdfUpload = async () => {
    if (!pdfFile) {
      setPdfError('Please select a PDF file');
      return;
    }

    setPdfProcessing(true);
    setPdfError('');
    setPdfResult(null);

    try {
      const response = await apiClient.uploadPdf(pdfFile, useAI, pdfCaption, pdfKeywords);
      setPdfResult(response);
      
      // Clear the file input to prevent duplicate processing
      setPdfFile(null);
      
      console.log('PDF Upload Response:', response);
      console.log('Keywords by method:', response.keywords_by_method);
      console.log('Smart caption:', response.smart_caption);
      
      // Instead of auto-indexing, show review modal for user to edit metadata
      if (response.success && response.images_extracted > 0) {
        setReviewData(response);
        setEditedCaption(response.smart_caption || pdfCaption || '');
        
        // Combine user keywords + suggested keywords
        const userKws = pdfKeywords.split(',').map(k => k.trim()).filter(k => k);
        const allKeywords = [...userKws, ...(response.suggested_keywords || [])];
        // Remove duplicates
        const unique = Array.from(new Set(allKeywords.map(k => k.toLowerCase()))).map(k => 
          allKeywords.find(kw => kw.toLowerCase() === k) || k
        );
        setEditedKeywords(unique);
        setKeywordsByMethod(response.keywords_by_method || {});
        setShowReviewModal(true);
      }
    } catch (err: any) {
      setPdfError(err.response?.data?.detail || 'PDF processing failed');
    } finally {
      setPdfProcessing(false);
    }
  };

  const handlePdfFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setPdfFile(e.target.files[0]);
      setPdfError('');
      setPdfResult(null); // Clear previous PDF upload result when selecting new file
      setPdfCaption(''); // Clear caption input
      setPdfKeywords(''); // Clear keywords input
    }
  };

  const handleConfirmMetadata = async () => {
    if (!reviewData) return;
    
    setLoading(true);
    try {
      // Save metadata with edited values
      await apiClient.savePdfMetadata(
        reviewData.image_paths,
        editedCaption,
        editedKeywords,
        reviewData.pdf_path.split('/').pop() || 'unknown.pdf'
      );
      
      setShowReviewModal(false);
      
      // Now trigger indexing
      setTimeout(() => {
        handleIndex();
      }, 500);
    } catch (err: any) {
      setPdfError(err.response?.data?.detail || 'Failed to save metadata');
    } finally {
      setLoading(false);
    }
  };

  const toggleKeyword = (keyword: string) => {
    setEditedKeywords(prev => 
      prev.includes(keyword)
        ? prev.filter(k => k !== keyword)
        : [...prev, keyword]
    );
  };

  const addCustomKeyword = (keyword: string) => {
    const trimmed = keyword.trim();
    if (trimmed && !editedKeywords.includes(trimmed)) {
      setEditedKeywords(prev => [...prev, trimmed]);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Image Upload Section */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-6 flex items-center">
          <Upload className="w-7 h-7 mr-3 text-blue-600" />
          Upload Images
        </h2>

        <div className="space-y-6">
          <div>
            <label htmlFor="imageFiles" className="block text-sm font-medium text-gray-700 mb-2">
              Select Images
            </label>
            <input
              id="imageFiles"
              type="file"
              accept="image/*"
              multiple
              onChange={handleImageFileChange}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg text-gray-900 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
            <p className="mt-2 text-sm text-gray-500">
              {imageFiles.length > 0 
                ? `Selected: ${imageFiles.length} image(s)` 
                : 'Select one or more images to upload to the server'}
            </p>
          </div>

          {imageFiles.length > 0 && (
            <div className="border border-gray-200 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Selected Images:</h4>
              <div className="space-y-2 max-h-40 overflow-y-auto">
                {imageFiles.map((file, idx) => (
                  <div key={idx} className="flex items-center justify-between text-sm text-gray-600 bg-gray-50 px-3 py-2 rounded">
                    <span className="truncate flex-1">{file.name}</span>
                    <button
                      onClick={() => removeImage(idx)}
                      className="ml-2 text-red-600 hover:text-red-800"
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Caption (optional)</label>
                  <input
                    type="text"
                    value={imageCaption}
                    onChange={(e) => setImageCaption(e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg text-gray-900"
                    placeholder="Enter caption or leave blank"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Keywords (optional)</label>
                  <input
                    type="text"
                    value={imageKeywords}
                    onChange={(e) => setImageKeywords(e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg text-gray-900"
                    placeholder="comma,separated,keywords"
                  />
                  <p className="text-xs text-gray-500 mt-1">Separate keywords with commas</p>
                </div>
              </div>
            </div>
          )}

          {uploadError && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start">
              <AlertCircle className="w-5 h-5 text-red-600 mr-3 flex-shrink-0 mt-0.5" />
              <div className="text-red-800">{uploadError}</div>
            </div>
          )}

          {uploadResult && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="flex items-start">
                <CheckCircle className="w-5 h-5 text-green-600 mr-3 flex-shrink-0 mt-0.5" />
                <div className="space-y-2">
                  <p className="text-green-800 font-medium">
                    {uploadResult.message}
                  </p>
                  <div className="text-sm text-green-700">
                    {/* <p>
                      {`Uploaded ${lastUploadedCount} image${lastUploadedCount === 1 ? '' : 's'}. Indexing will start automatically.`}
                    </p> */}
                  </div>
                </div>
              </div>
            </div>
          )}

          <button
            onClick={handleImageUpload}
            disabled={uploadingImages || imageFiles.length === 0}
            className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
          >
            {uploadingImages ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Uploading...</span>
              </>
            ) : (
              <>
                <Upload className="w-5 h-5" />
                <span>Upload Images</span>
              </>
            )}
          </button>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="font-medium text-blue-900 mb-2">Upload Process</h3>
            <ol className="text-sm text-blue-800 space-y-1 list-decimal list-inside">
              <li>Select one or more images from your computer</li>
              <li>Click "Upload Images" to send them to the server</li>
              <li>Images will be saved to the server's images folder</li>
              <li>Automatic indexing will start after upload completes</li>
            </ol>
          </div>
        </div>
      </div>

      {/* PDF Image Extraction Section */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-6 flex items-center">
          <Upload className="w-7 h-7 mr-3 text-green-600" />
          Extract Images from PDF
        </h2>

        <div className="space-y-6">
          <div>
            <label htmlFor="pdfFile" className="block text-sm font-medium text-gray-700 mb-2">
              Upload PDF
            </label>
            <input
              id="pdfFile"
              type="file"
              accept=".pdf"
              onChange={handlePdfFileChange}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg text-gray-900 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-green-50 file:text-green-700 hover:file:bg-green-100"
            />
            <p className="mt-2 text-sm text-gray-500">
              {pdfFile ? `Selected: ${pdfFile.name}` : 'Upload a PDF to extract images and metadata'}
            </p>
          </div>

          <div className="flex items-center space-x-2">
            <input
              id="useAI"
              type="checkbox"
              checked={useAI}
              onChange={(e) => setUseAI(e.target.checked)}
              className="w-4 h-4 text-green-600 border-gray-300 rounded focus:ring-green-500"
            />
            <label htmlFor="useAI" className="text-sm text-gray-700">
              Use AI analysis (requires Ollama running locally)
            </label>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Caption for extracted images (optional)</label>
              <input
                type="text"
                value={pdfCaption}
                onChange={(e) => setPdfCaption(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg text-gray-900"
                placeholder="Will use AI smart caption if left blank"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Keywords for extracted images (optional)</label>
              <input
                type="text"
                value={pdfKeywords}
                onChange={(e) => setPdfKeywords(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg text-gray-900"
                placeholder="comma,separated,keywords"
              />
              <p className="text-xs text-gray-500 mt-1">Separate keywords with commas</p>
            </div>
          </div>

          {pdfError && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start">
              <AlertCircle className="w-5 h-5 text-red-600 mr-3 flex-shrink-0 mt-0.5" />
              <div className="text-red-800">{pdfError}</div>
            </div>
          )}

          {pdfResult && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="flex items-start">
                <CheckCircle className="w-5 h-5 text-green-600 mr-3 flex-shrink-0 mt-0.5" />
                <div className="space-y-2 flex-1">
                  <p className="text-green-800 font-medium">
                    {pdfResult.message}
                  </p>
                  <div className="text-sm text-green-700 space-y-1">
                    <p>
                      {`Images extracted: ${pdfResult.images_extracted}. Indexing will start automatically.`}
                    </p>
                    <p>PDF saved to: {pdfResult.pdf_path}</p>
                    
                    {pdfResult.suggested_keywords && pdfResult.suggested_keywords.length > 0 && (
                      <div className="mt-2">
                        <p className="font-medium">Suggested Keywords (from 4 methods):</p>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {pdfResult.suggested_keywords.map((keyword: string, idx: number) => (
                            <span key={idx} className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">
                              {keyword}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {pdfResult.ai_info && !pdfResult.ai_info.error && (
                      <div className="mt-3 p-3 bg-white rounded border border-green-200">
                        <p className="font-medium text-green-900 mb-2">AI Analysis Results:</p>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          {Object.entries(pdfResult.ai_info).map(([key, value]: [string, any]) => (
                            <div key={key}>
                              <span className="font-medium">{key.replace(/_/g, ' ')}:</span>{' '}
                              <span className="text-gray-700">{value || 'N/A'}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {pdfResult.smart_caption && (
                      <div className="mt-2">
                        <p className="font-medium">Smart Caption:</p>
                        <p className="text-gray-700 italic">{pdfResult.smart_caption}</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          <button
            onClick={handlePdfUpload}
            disabled={pdfProcessing || !pdfFile}
            className="w-full bg-green-600 text-white py-3 px-6 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
          >
            {pdfProcessing ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Processing PDF...</span>
              </>
            ) : (
              <>
                <Upload className="w-5 h-5" />
                <span>Extract Images & Analyze</span>
              </>
            )}
          </button>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="font-medium text-blue-900 mb-2">PDF Processing Steps</h3>
            <ol className="text-sm text-blue-800 space-y-1 list-decimal list-inside">
              <li>Extract and save all embedded images from PDF</li>
              <li>Save PDF to catalog folder for reference</li>
              <li>Analyze content using 4 methods:
                <ul className="ml-6 mt-1 space-y-0.5 list-disc">
                  <li><strong>AI Analysis</strong>: Extract structured info (project, client, location, etc.)</li>
                  <li><strong>Pattern-based</strong>: Find labeled fields like "Project:", "Location:"</li>
                  <li><strong>Capitalized phrases</strong>: Extract proper nouns and names</li>
                  <li><strong>Frequency analysis</strong>: Find most common meaningful words</li>
                </ul>
              </li>
              <li>Generate smart captions and keywords for images</li>
              <li>Ready for indexing (use button below)</li>
            </ol>
          </div>
        </div>
      </div>

      {/* Index Server Images Section */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h1 className="text-3xl font-bold mb-6 flex items-center">
          <FolderOpen className="w-8 h-8 mr-3 text-purple-600" />
          Index Server Images
        </h1>

        <div className="space-y-6">
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <p className="text-sm text-purple-800">
              <strong>Server Image Folder:</strong> C:\AIIMGS\images
            </p>
            <p className="text-sm text-purple-700 mt-2">
              Click the button below to index all images in the server folder. 
              Only new or changed images will be processed (incremental indexing).
            </p>
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start">
              <AlertCircle className="w-5 h-5 text-red-600 mr-3 flex-shrink-0 mt-0.5" />
              <div className="text-red-800">{error}</div>
            </div>
          )}

          {result && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="flex items-start">
                <CheckCircle className="w-5 h-5 text-green-600 mr-3 flex-shrink-0 mt-0.5" />
                <div className="space-y-2">
                  <p className="text-green-800 font-medium">
                    Indexing completed successfully!
                  </p>
                  <div className="text-sm text-green-700">
                    <p className="font-medium">{result.message}</p>
                    <p>Duration: {result.duration?.toFixed(2) || 0} seconds</p>
                    <p>Model: {result.model_name || 'clip-vit-b-16'} (from config.json)</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          <button
            onClick={handleIndex}
            disabled={loading}
            className="w-full bg-purple-600 text-white py-3 px-6 rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Indexing in progress...</span>
              </>
            ) : (
              <>
                <FolderOpen className="w-5 h-5" />
                <span>Start Indexing</span>
              </>
            )}
          </button>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="font-medium text-blue-900 mb-2">About Indexing</h3>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• <strong>Incremental indexing</strong>: Only processes new or changed images</li>
              <li>• Automatically detects and removes deleted images from index</li>
              <li>• Preserves existing embeddings to save time and resources</li>
              <li>• The CLIP model is configured in config.json (currently: ViT-B/16)</li>
              <li>• Images must be indexed before they can be searched</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Review Modal for PDF Extracted Images */}
      {showReviewModal && reviewData && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={() => setShowReviewModal(false)}>
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="p-6 space-y-6">
              <div className="flex justify-between items-start">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">Review & Edit Metadata</h2>
                  <p className="text-sm text-gray-600 mt-1">
                    {reviewData.images_extracted} images extracted from PDF
                  </p>
                </div>
                <button
                  onClick={() => setShowReviewModal(false)}
                  className="text-gray-500 hover:text-gray-700 text-2xl"
                >
                  ×
                </button>
              </div>

              {/* Caption Editor */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Caption for all extracted images
                </label>
                <input
                  type="text"
                  value={editedCaption}
                  onChange={(e) => setEditedCaption(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg text-gray-900"
                  placeholder="Enter a caption..."
                />
                {reviewData.smart_caption && (
                  <p className="text-xs text-gray-500 mt-1">
                    AI suggested: "{reviewData.smart_caption}"
                  </p>
                )}
              </div>

              {/* Keywords by Method */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Keywords (click to toggle)
                </label>
                
                <div className="space-y-4">
                  {Object.entries(keywordsByMethod)
                    .filter(([_, keywords]) => keywords && keywords.length > 0)
                    .map(([method, keywords]: [string, any]) => (
                      <div key={method} className="mb-4">
                        <p className="text-xs font-medium text-gray-600 mb-2">{method}:</p>
                        <div className="flex flex-wrap gap-2">
                          {keywords.map((keyword: string) => (
                            <button
                              key={keyword}
                              onClick={() => toggleKeyword(keyword)}
                              className={`px-3 py-1 rounded-full text-sm transition-colors ${
                                editedKeywords.includes(keyword)
                                  ? 'bg-blue-600 text-white'
                                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                              }`}
                            >
                              {keyword}
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                </div>

                {/* Add custom keyword */}
                <div className="mt-4">
                  <input
                    type="text"
                    placeholder="Add custom keyword and press Enter..."
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg text-gray-900"
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        addCustomKeyword(e.currentTarget.value);
                        e.currentTarget.value = '';
                      }
                    }}
                  />
                </div>

                {/* Selected keywords display */}
                {editedKeywords.length > 0 && (
                  <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                    <p className="text-sm font-medium text-blue-900 mb-2">
                      Selected Keywords ({editedKeywords.length}):
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {editedKeywords.map((keyword) => (
                        <span
                          key={keyword}
                          className="px-2 py-1 bg-blue-600 text-white rounded text-sm flex items-center gap-1"
                        >
                          {keyword}
                          <button
                            onClick={() => toggleKeyword(keyword)}
                            className="hover:text-red-200"
                          >
                            ×
                          </button>
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="flex justify-end space-x-3 pt-4 border-t">
                <button
                  onClick={() => setShowReviewModal(false)}
                  className="px-6 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 text-gray-700"
                >
                  Cancel
                </button>
                <button
                  onClick={handleConfirmMetadata}
                  disabled={loading}
                  className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 flex items-center space-x-2"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Saving...</span>
                    </>
                  ) : (
                    <>
                      <CheckCircle className="w-4 h-4" />
                      <span>Confirm & Index</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
