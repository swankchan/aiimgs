/**
 * API client for AI Image Search backend
 */
import axios, { AxiosInstance, AxiosError } from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface SearchResult {
  path: string;
  score: number;
  caption?: string;
  keywords?: string[];
  thumbnail_url?: string;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  duration: number;
}

export interface User {
  id: number;
  username: string;
  full_name?: string;
  email?: string;
  is_admin: boolean;
  created_at?: string;
  last_login?: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface ImageMetadata {
  path: string;
  caption: string;
  keywords: string[];
}

export interface LibraryItem {
  path: string;
  type: string;
  name: string;
  caption?: string;
  keywords?: string[];
  thumbnail_url?: string;
}

export interface LibraryResponse {
  items: LibraryItem[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

export interface IndexStats {
  total_images: number;
  total_pdfs: number;
  index_size: number;
  last_updated?: string;
}

class ApiClient {
  private client: AxiosInstance;
  private token: string | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add interceptor to include auth token
    this.client.interceptors.request.use((config) => {
      if (this.token) {
        config.headers.Authorization = `Bearer ${this.token}`;
      }
      return config;
    });

    // Load token from localStorage on init
    if (typeof window !== 'undefined') {
      this.token = localStorage.getItem('auth_token');
    }
  }

  setToken(token: string | null) {
    this.token = token;
    if (typeof window !== 'undefined') {
      if (token) {
        localStorage.setItem('auth_token', token);
      } else {
        localStorage.removeItem('auth_token');
      }
    }
  }

  getToken(): string | null {
    return this.token;
  }

  // Authentication
  async login(username: string, password: string): Promise<LoginResponse> {
    const response = await this.client.post<LoginResponse>('/api/auth/login', {
      username,
      password,
    });
    this.setToken(response.data.access_token);
    return response.data;
  }

  async logout() {
    this.setToken(null);
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.client.get<User>('/api/auth/me');
    return response.data;
  }

  // Search
  async searchByText(query: string, topK: number = 32): Promise<SearchResponse> {
    const response = await this.client.post<SearchResponse>('/api/search/text', {
      query,
      top_k: topK,
    });
    return response.data;
  }

  async searchByImage(file: File, topK: number = 32): Promise<SearchResponse> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await this.client.post<SearchResponse>(
      `/api/search/image?top_k=${topK}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  }

  async getSearchStats() {
    const response = await this.client.get('/api/search/stats');
    return response.data;
  }

  // Indexing
  async indexFolder(folderPath: string, modelName: string = 'clip-vit-b-32') {
    const response = await this.client.post('/api/indexing/index', {
      folder_path: folderPath,
      model_name: modelName,
    });
    return response.data;
  }

  async syncFolders(folders: string[]) {
    const response = await this.client.post('/api/indexing/sync', { folders });
    return response.data;
  }

  async uploadImages(files: File[], caption?: string, keywords?: string) {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    if (caption !== undefined) formData.append('caption', caption || '');
    if (keywords !== undefined) formData.append('keywords', keywords || '');
    
    const response = await this.client.post('/api/indexing/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async uploadPdf(file: File, useAi: boolean = false, caption?: string, keywords?: string) {
    const formData = new FormData();
    formData.append('file', file);
    if (caption !== undefined) formData.append('caption', caption || '');
    if (keywords !== undefined) formData.append('keywords', keywords || '');
    
    const response = await this.client.post(
      `/api/indexing/upload-pdf?use_ai=${useAi}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  }

  async removeImages(paths: string[], deleteFiles: boolean = false) {
    const response = await this.client.delete('/api/indexing/remove', {
      params: { delete_files: deleteFiles },
      data: { paths },
    });
    return response.data;
  }

  async getIndexStats(): Promise<IndexStats> {
    const response = await this.client.get<IndexStats>('/api/indexing/stats');
    return response.data;
  }

  async savePdfMetadata(imagePaths: string[], caption: string, keywords: string[], pdfFilename: string) {
    const response = await this.client.post('/api/indexing/save-pdf-metadata', {
      image_paths: imagePaths,
      caption,
      keywords,
      pdf_filename: pdfFilename
    });
    return response.data;
  }

  // Metadata
  async getAllMetadata() {
    const response = await this.client.get('/api/metadata/all');
    return response.data;
  }

  async getMetadata(path: string): Promise<ImageMetadata> {
    const response = await this.client.get<ImageMetadata>(`/api/metadata/${encodeURIComponent(path)}`);
    return response.data;
  }

  async saveMetadata(metadata: ImageMetadata) {
    const response = await this.client.post('/api/metadata/', metadata);
    return response.data;
  }

  async extractMetadataFromPdf(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await this.client.post('/api/metadata/extract-pdf', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async updateMetadata(path: string, data: { caption?: string; keywords?: string[] }) {
    const response = await this.client.patch(`/api/metadata/${encodeURIComponent(path)}`, data);
    return response.data;
  }

  // Library
  async getLibrary(
    page: number = 1,
    perPage: number = 1000
  ) {
    const response = await this.client.get('/api/library/', {
      params: { page, per_page: perPage },
    });
    return response.data;
  }

  async listFolders() {
    const response = await this.client.get('/api/library/folders');
    return response.data;
  }

  // Admin
  async getUsers(): Promise<User[]> {
    const response = await this.client.get<User[]>('/api/admin/users');
    return response.data;
  }

  async createUser(userData: {
    username: string;
    password: string;
    full_name?: string;
    email?: string;
    is_admin: boolean;
  }): Promise<User> {
    const response = await this.client.post<User>('/api/admin/users', userData);
    return response.data;
  }

  async updateUser(userId: number, userData: {
    full_name?: string;
    email?: string;
    is_admin?: boolean;
    password?: string;
  }): Promise<User> {
    const response = await this.client.put<User>(`/api/admin/users/${userId}`, userData);
    return response.data;
  }

  async deleteUser(userId: number): Promise<void> {
    await this.client.delete(`/api/admin/users/${userId}`);
  }

  // Health check
  async healthCheck() {
    const response = await this.client.get('/api/health');
    return response.data;
  }
}

export const apiClient = new ApiClient();
