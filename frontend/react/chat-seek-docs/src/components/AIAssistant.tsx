import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { DatabaseSection } from '@/components/DatabaseSection';
import { PDFChatSection } from '@/components/PDFChatSection';
import { MessageCircle, BarChart3 } from 'lucide-react';

export const AIAssistant = () => {
  const [activeTab, setActiveTab] = useState('chat');

  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
              <MessageCircle className="h-5 w-5 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-lg font-medium text-slate-100">InsightPad</h1>
              <p className="text-sm text-muted-foreground">Chat with your data and get insights in real-time</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 container mx-auto px-4 py-6 min-h-0">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
          <TabsList className="grid w-full grid-cols-2 mb-6 bg-secondary/50">
            <TabsTrigger value="chat" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground flex items-center gap-2">
              <MessageCircle className="h-4 w-4" />
              Chat
            </TabsTrigger>
            <TabsTrigger value="analytics" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Analytics
            </TabsTrigger>
          </TabsList>

          <div className="flex-1 min-h-0">
            <TabsContent value="chat" className="h-full m-0">
              <PDFChatSection />
            </TabsContent>

            <TabsContent value="analytics" className="h-full m-0">
              <div className="bg-card rounded-lg border h-full">
                <DatabaseSection />
              </div>
            </TabsContent>
          </div>
        </Tabs>
      </div>
    </div>
  );
};