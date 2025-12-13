'use client';

import { useEffect, useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useAuthStore } from '@/lib/store';
import LoginPage from '@/components/LoginPage';
import Navbar from '@/components/Navbar';
import './globals.css';

const queryClient = new QueryClient();

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { isAuthenticated, checkAuth } = useAuthStore();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    checkAuth().finally(() => setIsLoading(false));
  }, [checkAuth]);

  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/img/icon.png" type="image/png" />
      </head>
      <body>
        <QueryClientProvider client={queryClient}>
          {isLoading ? (
            <div className="min-h-screen flex items-center justify-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            </div>
          ) : !isAuthenticated ? (
            <LoginPage />
          ) : (
            <>
              <Navbar />
              <main className="container mx-auto px-4 py-8">
                {children}
              </main>
            </>
          )}
        </QueryClientProvider>
      </body>
    </html>
  );
}
