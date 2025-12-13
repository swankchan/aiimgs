'use client';

import { useState } from 'react';
import { SearchResult } from '@/lib/api';

interface ImageGridProps {
  results: SearchResult[];
  itemsPerPage?: number;
}

export default function ImageGrid({ results, itemsPerPage = 16 }: ImageGridProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const [perPage, setPerPage] = useState(itemsPerPage);
  const [selectedImage, setSelectedImage] = useState<SearchResult | null>(null);

  const totalPages = Math.ceil(results.length / perPage);
  const startIdx = (currentPage - 1) * perPage;
  const endIdx = startIdx + perPage;
  const pageResults = results.slice(startIdx, endIdx);

  const handlePerPageChange = (newPerPage: number) => {
    setPerPage(newPerPage);
    setCurrentPage(1); // Reset to first page
  };

  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  return (
    <div>
      {/* Pagination controls top */}
      {results.length > 0 && (
        <div className="mb-6 flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-700">
              Showing {Math.min(startIdx + 1, results.length)} - {Math.min(endIdx, results.length)} of {results.length} images
            </span>
            <div className="flex items-center space-x-2">
              <label className="text-sm text-gray-700">Per page:</label>
              <select
                value={perPage}
                onChange={(e) => handlePerPageChange(Number(e.target.value))}
                className="border border-gray-300 rounded px-2 py-1 text-sm text-gray-900"
              >
                <option value={10}>10</option>
                <option value={16}>16</option>
                <option value={20}>20</option>
                <option value={32}>32</option>
                <option value={50}>50</option>
              </select>
            </div>
          </div>

          {totalPages > 1 && (
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
          )}
        </div>
      )}

      {/* Image grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {pageResults.map((result, idx) => (
          <div
            key={`${result.path}-${idx}`}
            className="relative group cursor-pointer"
            onClick={() => setSelectedImage(result)}
          >
            <img
              src={result.thumbnail_url ? `http://localhost:8000${result.thumbnail_url}` : `http://localhost:8000/images/${result.path.split('\\').pop()}`}
              alt={result.caption || 'Image'}
              className="square-image hover:opacity-90 transition-opacity"
            />
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-2 rounded-b-lg opacity-0 group-hover:opacity-100 transition-opacity">
              <p className="text-white text-sm truncate">
                {result.caption || result.path.split('/').pop()}
              </p>
              <p className="text-white/80 text-xs">
                Score: {result.score.toFixed(2)}
              </p>
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

      {/* Image modal */}
      {selectedImage && (
        <div
          className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4"
          onClick={() => setSelectedImage(null)}
        >
          <div
            className="bg-white rounded-lg max-w-4xl max-h-[90vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-xl font-bold">Image Details</h3>
                <button
                  onClick={() => setSelectedImage(null)}
                  className="text-gray-500 hover:text-gray-700 text-2xl"
                >
                  ×
                </button>
              </div>
              
              <img
                src={selectedImage.thumbnail_url ? `http://localhost:8000${selectedImage.thumbnail_url}` : `http://localhost:8000/images/${selectedImage.path.split('\\').pop()}`}
                alt={selectedImage.caption || 'Image'}
                className="w-full rounded-lg mb-4"
              />
              
              <div className="space-y-2">
                <p className="text-sm text-gray-600">
                  <strong>Path:</strong> {selectedImage.path}
                </p>
                {selectedImage.caption && (
                  <p className="text-sm text-gray-600">
                    <strong>Caption:</strong> {selectedImage.caption}
                  </p>
                )}
                {Array.isArray(selectedImage.keywords) && selectedImage.keywords.length > 0 && (
                  <p className="text-sm text-gray-600">
                    <strong>Keywords:</strong> {selectedImage.keywords.join(', ')}
                  </p>
                )}
                <p className="text-sm text-gray-600">
                  <strong>Similarity Score:</strong> {selectedImage.score?.toFixed(3) || 'N/A'}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
