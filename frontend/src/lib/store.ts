/**
 * Global state management using Zustand
 */
import { create } from 'zustand';
import { apiClient, User } from './api';

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  checkAuth: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  isAuthenticated: false,

  login: async (username: string, password: string) => {
    const response = await apiClient.login(username, password);
    set({ user: response.user, isAuthenticated: true });
  },

  logout: () => {
    apiClient.logout();
    set({ user: null, isAuthenticated: false });
  },

  checkAuth: async () => {
    try {
      if (apiClient.getToken()) {
        const user = await apiClient.getCurrentUser();
        set({ user, isAuthenticated: true });
      }
    } catch (error) {
      set({ user: null, isAuthenticated: false });
    }
  },
}));

interface SearchState {
  searchResults: any[];
  searchDuration: number;
  setSearchResults: (results: any[], duration: number) => void;
  clearSearchResults: () => void;
}

export const useSearchStore = create<SearchState>((set) => ({
  searchResults: [],
  searchDuration: 0,

  setSearchResults: (results, duration) => {
    set({ searchResults: results, searchDuration: duration });
  },

  clearSearchResults: () => {
    set({ searchResults: [], searchDuration: 0 });
  },
}));
