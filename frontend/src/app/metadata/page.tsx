'use client';

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import { apiClient } from '@/lib/api';
import { FileText, Save, Loader2, ArrowLeft } from 'lucide-react';
import Link from 'next/link';

export default function MetadataPage() {
  const searchParams = useSearchParams();
  const imagePathFromUrl = searchParams.get('image');
  
  const [imagePath, setImagePath] = useState('');
  const [caption, setCaption] = useState('');
  const [keywords, setKeywords] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');
  const [systemTags, setSystemTags] = useState<string[]>([]);

  useEffect(() => {
    if (imagePathFromUrl) {
      setImagePath(imagePathFromUrl);
      loadMetadataForPath(imagePathFromUrl);
    }
  }, [imagePathFromUrl]);

  const loadMetadataForPath = async (path: string) => {
    setLoading(true);
    setMessage('');

    try {
      const metadata = await apiClient.getMetadata(path);
      setCaption(metadata.caption || '');
      // Exclude read-only system tags from editable keywords input
      const systemPrefixes = ['origin_user:', 'origin_pdf:', 'uploaded_at:'];
      const allKeywords = metadata.keywords || [];
      const userKeywords = allKeywords.filter(
        (k) => !systemPrefixes.some((pref) => String(k).startsWith(pref))
      );
      const sysTags = allKeywords.filter(
        (k) => systemPrefixes.some((pref) => String(k).startsWith(pref))
      );
      setKeywords(userKeywords.join(', '));
      setSystemTags(sysTags);
      setMessage('Metadata loaded successfully');
    } catch (err: any) {
      setMessage(err.response?.data?.detail || 'Failed to load metadata');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveMetadata = async () => {
    if (!imagePath.trim()) {
      setMessage('No image selected');
      return;
    }

    setSaving(true);
    setMessage('');

    try {
      const keywordsArray = keywords
        .split(',')
        .map(k => k.trim())
        .filter(k => k.length > 0);

      // Always send both fields to allow clearing them
      await apiClient.updateMetadata(imagePath, {
        caption: caption.trim(),
        keywords: keywordsArray,
      });

      setMessage('Metadata saved successfully');
    } catch (err: any) {
      setMessage(err.response?.data?.detail || 'Failed to save metadata');
    } finally {
      setSaving(false);
    }
  };

  if (!imagePath) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h1 className="text-3xl font-bold mb-6 flex items-center">
            <FileText className="w-8 h-8 mr-3 text-blue-600" />
            Manage Metadata
          </h1>

          <div className="text-center py-12">
            <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-gray-700 mb-2">No Image Selected</h2>
            <p className="text-gray-600 mb-6">
              To edit metadata for an image, please select an image from the Library page.
            </p>
            <Link
              href="/library"
              className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <ArrowLeft className="w-5 h-5 mr-2" />
              Go to Library
            </Link>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mt-6">
            <h3 className="font-medium text-blue-900 mb-2">How to Edit Metadata</h3>
            <ol className="text-sm text-blue-800 space-y-1 list-decimal list-inside">
              <li>Go to the <strong>Library</strong> page</li>
              <li>Click on any image to view details</li>
              <li>Click "Edit Metadata" button to come to this page</li>
              <li>Edit caption and keywords, then save</li>
            </ol>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-3xl font-bold flex items-center">
            <FileText className="w-8 h-8 mr-3 text-blue-600" />
            Manage Metadata
          </h1>
          <Link
            href="/library"
            className="text-gray-600 hover:text-gray-900 flex items-center"
          >
            <ArrowLeft className="w-5 h-5 mr-1" />
            Back to Library
          </Link>
        </div>

        <div className="space-y-6">
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
            <p className="text-sm text-gray-600 mb-1">Editing metadata for:</p>
            <p className="text-sm font-mono text-gray-900 break-all">{imagePath}</p>
          </div>

          {/* Preview image */}
          <div className="flex justify-center">
            <img
              src={`http://localhost:8000${imagePath.startsWith('/') ? '' : '/'}${imagePath}`}
              alt="Preview"
              className="max-h-64 rounded-lg shadow-md object-contain"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none';
              }}
            />
          </div>

          <div>
            <label htmlFor="caption" className="block text-sm font-medium text-gray-700 mb-2">
              Caption
            </label>
            <textarea
              id="caption"
              value={caption}
              onChange={(e) => setCaption(e.target.value)}
              placeholder="Enter a descriptive caption for this image"
              rows={3}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900"
            />
          </div>

          <div>
            <label htmlFor="keywords" className="block text-sm font-medium text-gray-700 mb-2">
              Keywords
            </label>
            <input
              id="keywords"
              type="text"
              value={keywords}
              onChange={(e) => setKeywords(e.target.value)}
              placeholder="keyword1, keyword2, keyword3"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900"
            />
            <p className="mt-2 text-sm text-gray-500">
              Separate keywords with commas
            </p>
            {/* Read-only system tags */}
            <div className="mt-3">
              <p className="text-sm font-medium text-gray-700 mb-1">System Tags (read-only)</p>
              <div className="flex flex-wrap gap-2">
                {systemTags.length === 0 ? (
                  <span className="text-xs text-gray-500">None</span>
                ) : (
                  systemTags.map((tag, idx) => (
                    <span key={idx} className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-xs font-mono">
                      {tag}
                    </span>
                  ))
                )}
              </div>
            </div>
          </div>

          {message && (
            <div className={`p-4 rounded-lg ${
              message.includes('success') 
                ? 'bg-green-50 text-green-800 border border-green-200' 
                : 'bg-red-50 text-red-800 border border-red-200'
            }`}>
              {message}
            </div>
          )}

          <button
            onClick={handleSaveMetadata}
            disabled={saving || loading}
            className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
          >
            {saving ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Saving...</span>
              </>
            ) : (
              <>
                <Save className="w-5 h-5" />
                <span>Save Metadata</span>
              </>
            )}
          </button>
          
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="font-medium text-blue-900 mb-2">Note</h3>
            <p className="text-sm text-blue-800">
              To extract metadata from PDFs, please use the <strong>Indexing</strong> tab where you can upload PDFs 
              and automatically extract images with AI-powered content analysis.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
