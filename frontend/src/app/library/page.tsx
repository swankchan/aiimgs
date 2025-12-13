'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { Search, Loader2, Image as ImageIcon } from 'lucide-react';
import { useRouter } from 'next/navigation';

export default function LibraryPage() {
  const router = useRouter();
  
  const [images, setImages] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredImages, setFilteredImages] = useState<any[]>([]);
  const [error, setError] = useState<string>('');
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(20);
  const [totalItems, setTotalItems] = useState(0);
  const [selectedType, setSelectedType] = useState<'image' | 'pdf' | null>(null);
  const [initialized, setInitialized] = useState(false);

  // Restore state from localStorage on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('libraryState');
      if (saved) {
        try {
          const state = JSON.parse(saved);
          setSelectedType(state.selectedType || null);
          setSearchTerm(state.searchTerm || '');
          setCurrentPage(state.currentPage || 1);
          setItemsPerPage(state.itemsPerPage || 20);
        } catch (e) {
          console.error('Failed to restore library state', e);
        }
      }
      setInitialized(true);
    }
  }, []);

  // Save state to localStorage whenever it changes
  useEffect(() => {
    if (initialized && typeof window !== 'undefined') {
      const state = {
        selectedType,
        searchTerm,
        currentPage,
        itemsPerPage
      };
      localStorage.setItem('libraryState', JSON.stringify(state));
    }
  }, [selectedType, searchTerm, currentPage, itemsPerPage, initialized]);

  useEffect(() => {
    if (selectedType && initialized) {
      loadLibrary();
    }
  }, [selectedType, initialized]); // Reload when type changes or initialized

  useEffect(() => {
    // Apply search filter on the images
    let result = images;
    
    // Filter by selected type
    if (selectedType) {
      result = result.filter(img => img.type === selectedType);
    }
    
    // Filter by search term
    if (searchTerm.trim()) {
      const term = searchTerm.toLowerCase();
      result = result.filter(img => 
        img.path.toLowerCase().includes(term) ||
        img.caption?.toLowerCase().includes(term) ||
        img.keywords?.some((k: string) => k.toLowerCase().includes(term))
      );
    }
    
    setFilteredImages(result);
    setTotalItems(result.length);
    // Don't reset page when type changes, only when search term changes
    if (searchTerm.trim()) {
      setCurrentPage(1);
    }
  }, [searchTerm, images, selectedType]);

  const loadLibrary = async () => {
    setLoading(true);
    setError('');
    try {
      // Load ALL items by fetching multiple pages (API max is 100 per page)
      let allItems: any[] = [];
      let page = 1;
      let totalPages = 1;
      
      do {
        const response = await apiClient.getLibrary(page, 100);
        console.log(`Fetching page ${page}:`, response.items?.length, 'items');
        
        if (!response || !response.items) {
          setError('Invalid response from API: ' + JSON.stringify(response));
          setLoading(false);
          return;
        }
        
        allItems = allItems.concat(response.items);
        totalPages = response.total_pages || 1;
        page++;
      } while (page <= totalPages);
      
      console.log('Total items loaded:', allItems.length);
      
      // Ensure all data is serializable
      const imageResults = allItems.map((img: any, index: number) => {
        try {
          return {
            path: String(img.path || ''),
            score: 1.0,
            caption: String(img.caption || ''),
            keywords: Array.isArray(img.keywords) ? img.keywords.map((k: any) => String(k)) : [],
            thumbnail_url: img.thumbnail_url || null,
            name: String(img.name || ''),
            type: String(img.type || 'image')
          };
        } catch (err) {
          console.error('Error processing item', index, img, err);
          return null;
        }
      }).filter((img: any) => img !== null);
      
      console.log('Processed images:', imageResults.length);
      setImages(imageResults);
      setFilteredImages(imageResults);
      setTotalItems(imageResults.length);
    } catch (err: any) {
      console.error('Failed to load library:', err);
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to load library';
      console.error('Error details:', JSON.stringify(err.response?.data, null, 2));
      // Ensure error is a string for display
      setError(typeof errorMsg === 'object' ? JSON.stringify(errorMsg, null, 2) : String(errorMsg));
    } finally {
      setLoading(false);
    }
  };

  const totalPages = Math.ceil(totalItems / itemsPerPage);

  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const handleItemsPerPageChange = (newItemsPerPage: number) => {
    setItemsPerPage(newItemsPerPage);
    setCurrentPage(1); // Reset to first page
  };

  const handleTypeChange = (type: 'image' | 'pdf') => {
    setSelectedType(type);
    setCurrentPage(1);
  };

  const handleSearchChange = (search: string) => {
    setSearchTerm(search);
  };

  const handleItemClick = (item: any) => {
    if (item.type === 'pdf') {
      const url = `http://localhost:8000${item.path.startsWith('/') ? '' : '/'}${item.path}`;
      window.open(url, '_blank');
    } else {
      router.push(`/metadata?image=${encodeURIComponent(item.path)}`);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h1 className="text-3xl font-bold mb-6 flex items-center">
          <ImageIcon className="w-8 h-8 mr-3 text-blue-600" />
          Image Library
        </h1>

        <div className="mb-6">
          {/* Type selector: require choice before loading */}
          <div className="flex items-center gap-2 mb-4">
            <button
              onClick={() => handleTypeChange('image')}
              className={`px-4 py-2 rounded-lg border ${selectedType === 'image' ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-900 border-gray-300 hover:bg-gray-50'}`}
            >
              Images
            </button>
            <button
              onClick={() => handleTypeChange('pdf')}
              className={`px-4 py-2 rounded-lg border ${selectedType === 'pdf' ? 'bg-purple-600 text-white border-purple-600' : 'bg-white text-gray-900 border-gray-300 hover:bg-gray-50'}`}
            >
              PDFs
            </button>
          </div>
          {!selectedType && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4 text-sm text-yellow-800">
              Please choose to view Images or PDFs.
            </div>
          )}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => handleSearchChange(e.target.value)}
              placeholder={selectedType ? "Filter by path, caption, or keywords..." : "Select Images or PDFs first to enable search"}
              disabled={!selectedType}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900 disabled:bg-gray-100 disabled:cursor-not-allowed"
            />
          </div>
        </div>

        {/* Pagination controls top */}
        {!loading && !error && totalItems > 0 && (
          <div className="mb-6 flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-700">
                Showing {Math.min((currentPage - 1) * itemsPerPage + 1, totalItems)} - {Math.min(currentPage * itemsPerPage, totalItems)} of {totalItems} images
              </span>
              <div className="flex items-center space-x-2">
                <label className="text-sm text-gray-700">Per page:</label>
                <select
                  value={itemsPerPage}
                  onChange={(e) => handleItemsPerPageChange(Number(e.target.value))}
                  className="border border-gray-300 rounded px-2 py-1 text-sm text-gray-900"
                >
                  <option value={10}>10</option>
                  <option value={20}>20</option>
                  <option value={50}>50</option>
                  <option value={100}>100</option>
                </select>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={() => handlePageChange(1)}
                disabled={currentPage === 1}
                className="px-3 py-1 border border-gray-300 rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100 text-sm text-gray-900 bg-white"
              >
                First
              </button>
              <button
                onClick={() => handlePageChange(currentPage - 1)}
                disabled={currentPage === 1}
                className="px-3 py-1 border border-gray-300 rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100 text-sm text-gray-900 bg-white"
              >
                Previous
              </button>
              <span className="text-sm text-gray-700">
                Page {currentPage} of {totalPages}
              </span>
              <button
                onClick={() => handlePageChange(currentPage + 1)}
                disabled={currentPage === totalPages}
                className="px-3 py-1 border border-gray-300 rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100 text-sm text-gray-900 bg-white"
              >
                Next
              </button>
              <button
                onClick={() => handlePageChange(totalPages)}
                disabled={currentPage === totalPages}
                className="px-3 py-1 border border-gray-300 rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100 text-sm text-gray-900 bg-white"
              >
                Last
              </button>
            </div>
          </div>
        )}

        {selectedType === null ? (
          <div className="text-center py-12">
            <ImageIcon className="w-16 h-16 mx-auto mb-4 text-gray-400" />
            <p className="text-gray-600">Select Images or PDFs to load the library.</p>
          </div>
        ) : loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
            <span className="ml-3 text-gray-600">Loading library...</span>
          </div>
        ) : error ? (
          <div className="text-center py-12">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 inline-block max-w-2xl">
              <p className="text-red-800 font-semibold mb-2">Error loading library:</p>
              <pre className="text-left text-sm text-red-700 overflow-auto">{typeof error === 'object' ? JSON.stringify(error, null, 2) : String(error)}</pre>
            </div>
          </div>
        ) : filteredImages.length > 0 ? (
          <div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              {filteredImages
                .slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage)
                .map((img, idx) => (
                <div 
                  key={idx} 
                  className="border rounded-lg overflow-hidden cursor-pointer hover:shadow-lg transition-shadow group"
                  onClick={() => handleItemClick(img)}
                >
                  <div className="relative">
                    <img 
                      src={img.type === 'pdf'
                        ? `http://localhost:8000${img.thumbnail_url || '/catalog/' + (img.name || 'preview') + '.png'}`
                        : `http://localhost:8000${img.thumbnail_url || '/images/' + img.path.split('\\').pop()}`}
                      alt={img.name || (img.type === 'pdf' ? 'PDF' : 'Image')}
                      className="w-full h-48 object-cover group-hover:opacity-90 transition-opacity"
                    />
                    <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-10 transition-all flex items-center justify-center">
                      <span className="text-white opacity-0 group-hover:opacity-100 text-sm font-medium">
                        {img.type === 'pdf' ? 'Open PDF' : 'Edit Metadata'}
                      </span>
                    </div>
                  </div>
                  <div className="p-2">
                    <p className="text-xs truncate">{img.name}</p>
                    {img.caption && (
                      <p className="text-xs text-gray-500 truncate mt-1">{img.caption}</p>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Pagination controls bottom */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center space-x-2 mt-6">
                <button
                  onClick={() => handlePageChange(1)}
                  disabled={currentPage === 1}
                  className="px-4 py-2 border border-gray-300 rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100 text-gray-900 bg-white"
                >
                  First
                </button>
                <button
                  onClick={() => handlePageChange(currentPage - 1)}
                  disabled={currentPage === 1}
                  className="px-4 py-2 border border-gray-300 rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100 text-gray-900 bg-white"
                >
                  Previous
                </button>
                <span className="text-gray-700">
                  Page {currentPage} of {totalPages}
                </span>
                <button
                  onClick={() => handlePageChange(currentPage + 1)}
                  disabled={currentPage === totalPages}
                  className="px-4 py-2 border border-gray-300 rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100 text-gray-900 bg-white"
                >
                  Next
                </button>
                <button
                  onClick={() => handlePageChange(totalPages)}
                  disabled={currentPage === totalPages}
                  className="px-4 py-2 border border-gray-300 rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100 text-gray-900 bg-white"
                >
                  Last
                </button>
              </div>
            )}
          </div>
        ) : images.length > 0 && selectedType ? (
          <div className="text-center py-12 text-gray-600">
            No items match your filter. Try a different search term.
          </div>
        ) : (
          <div className="text-center py-12">
            <ImageIcon className="w-16 h-16 mx-auto mb-4 text-gray-400" />
            <p className="text-gray-600">No items in library yet.</p>
            <p className="text-sm text-gray-500 mt-2">
              Index some images first to see them here.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
