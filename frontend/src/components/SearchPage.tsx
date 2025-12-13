'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api';
import { useSearchStore } from '@/lib/store';
import { Search, Upload, Loader2 } from 'lucide-react';
import ImageGrid from './ImageGrid';

export default function SearchPage() {
  const [searchMode, setSearchMode] = useState<'text' | 'image'>('text');
  const [textQuery, setTextQuery] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const { searchResults, searchDuration, setSearchResults, clearSearchResults } = useSearchStore();

  const handleTextSearch = async () => {
    if (!textQuery.trim()) return;
    
    setLoading(true);
    try {
      const response = await apiClient.searchByText(textQuery, 32);
      setSearchResults(response.results, response.duration);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleImageSearch = async () => {
    if (!imageFile) return;
    
    setLoading(true);
    try {
      const response = await apiClient.searchByImage(imageFile, 32);
      setSearchResults(response.results, response.duration);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleImageFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImageFile(e.target.files[0]);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4">Search Images</h2>

        {/* Search mode selector */}
        <div className="flex space-x-4 mb-6">
          <button
            onClick={() => {
              setSearchMode('text');
              clearSearchResults();
            }}
            className={`px-4 py-2 rounded-lg transition-colors ${
              searchMode === 'text'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            <Search className="w-4 h-4 inline mr-2" />
            Text Search
          </button>
          <button
            onClick={() => {
              setSearchMode('image');
              clearSearchResults();
            }}
            className={`px-4 py-2 rounded-lg transition-colors ${
              searchMode === 'image'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            <Upload className="w-4 h-4 inline mr-2" />
            Image Search
          </button>
        </div>

        {/* Text search */}
        {searchMode === 'text' && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Describe what you're looking for
              </label>
              <input
                type="text"
                value={textQuery}
                onChange={(e) => setTextQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleTextSearch()}
                placeholder="e.g., glass roof, beach, sunset..."
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900"
              />
            </div>
            <button
              onClick={handleTextSearch}
              disabled={loading || !textQuery.trim()}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Searching...</span>
                </>
              ) : (
                <>
                  <Search className="w-4 h-4" />
                  <span>Search</span>
                </>
              )}
            </button>
          </div>
        )}

        {/* Image search */}
        {searchMode === 'image' && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload an image to find similar ones
              </label>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageFileChange}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
              />
            </div>
            {imageFile && (
              <div className="mt-4">
                <img
                  src={URL.createObjectURL(imageFile)}
                  alt="Preview"
                  className="w-64 h-64 object-cover rounded-lg"
                />
              </div>
            )}
            <button
              onClick={handleImageSearch}
              disabled={loading || !imageFile}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Searching...</span>
                </>
              ) : (
                <>
                  <Search className="w-4 h-4" />
                  <span>Search</span>
                </>
              )}
            </button>
          </div>
        )}
      </div>

      {/* Search results */}
      {searchResults.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-bold">
              Found {searchResults.length} results
            </h3>
            <p className="text-sm text-gray-600">
              Search took {(searchDuration * 1000).toFixed(1)} ms
            </p>
          </div>
          <ImageGrid results={searchResults} />
        </div>
      )}

      {!loading && searchResults.length === 0 && (textQuery || imageFile) && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
          <p className="text-yellow-800">
            No results found. Try a different query or make sure images are indexed.
          </p>
        </div>
      )}
    </div>
  );
}
