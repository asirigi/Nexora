import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { chatAPI } from '@/lib/api';
import { Message } from '@/types';
import { FileText, Send, Loader2, Upload, X, Paperclip } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';

export const PDFChatSection = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const { toast } = useToast();
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFileUpload = async (file: File) => {
    if (file.type !== 'application/pdf') {
      toast({
        title: "Invalid file type",
        description: "Please upload a PDF file.",
        variant: "destructive",
      });
      return;
    }
    try {
      setIsLoading(true);
      const res = await chatAPI.uploadPdf(file);
      if (!res.success) throw new Error(res.error || 'Upload failed');
      setUploadedFile(file);
      setMessages([]);
      toast({
        title: "PDF uploaded successfully",
        description: `${file.name} is ready for questions.`,
      });
    } catch (err: any) {
      toast({
        title: "Upload failed",
        description: err.message || "Could not upload PDF",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const removeFile = () => {
    setUploadedFile(null);
    setMessages([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || !uploadedFile) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input.trim(),
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await chatAPI.pdfChat(userMessage.content);
      if (!response.success) throw new Error(response.error);
      const data: any = response.data;
      const content = typeof data === 'string' ? data : (data?.answer || data?.message || '');
      if (!content) throw new Error('Empty response from server');
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content,
        sender: 'assistant',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error: any) {
      toast({
        title: "PDF Chat Error",
        description: error.message || "Failed to process PDF chat",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const PDFMessage = ({ message }: { message: Message }) => {
    const isUser = message.sender === 'user';
    
    return (
      <div className={cn("flex gap-3 p-3", isUser ? "flex-row-reverse" : "flex-row")}>
        <div
          className={cn(
            "flex h-7 w-7 shrink-0 select-none items-center justify-center rounded-lg text-xs",
            isUser
              ? "bg-primary text-primary-foreground"
              : "bg-pdf-chat text-pdf-chat-foreground"
          )}
        >
          {isUser ? "U" : <FileText className="h-3 w-3" />}
        </div>
        
        <div className={cn("flex flex-col space-y-1 text-sm max-w-sm", isUser ? "items-end" : "items-start")}>
          <div
            className={cn(
              "px-3 py-2 rounded-xl text-sm",
              isUser
                ? "bg-primary text-primary-foreground rounded-br-sm"
                : "bg-pdf-panel text-foreground rounded-bl-sm border border-pdf-chat/20"
            )}
          >
            <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
          </div>
          
          <span className="text-xs text-muted-foreground px-2">
            {message.timestamp.toLocaleTimeString()}
          </span>
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto h-full flex flex-col">
      <div className="flex-1 flex flex-col min-h-0">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-semibold text-foreground mb-2">Chat with your data</h1>
          <p className="text-muted-foreground">Ask questions about your PDF documents and get insights in real-time.</p>
        </div>

        {/* Chat Input */}
        <div className="mb-8">
          <form onSubmit={handleSendMessage} className="flex gap-3">
            <div className="flex-1 relative">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a question about your PDF..."
                disabled={isLoading || !uploadedFile}
                className="pl-12 h-12 text-base bg-secondary/50 border-0"
              />
              <Paperclip className="absolute left-4 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            </div>
            <Button 
              type="submit" 
              size="lg"
              disabled={!input.trim() || isLoading || !uploadedFile}
              className="px-8"
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                "Send"
              )}
            </Button>
          </form>
        </div>

        {/* Data Sources Section */}
        <div className="mb-8">
          <h2 className="text-xl font-medium text-foreground mb-4">Data Sources</h2>
          <div className="space-y-4">
            {/* Upload Files */}
            <Card className="cursor-pointer hover:bg-secondary/50 transition-colors"
                  onClick={() => fileInputRef.current?.click()}>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-secondary flex items-center justify-center">
                    <Upload className="h-5 w-5 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="font-medium">Upload Files</p>
                    <p className="text-sm text-muted-foreground">PDF documents only</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Uploaded Files Display */}
            {uploadedFile && (
              <Card className="bg-primary/5 border-primary/20">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                        <FileText className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <p className="font-medium">{uploadedFile.name}</p>
                        <p className="text-sm text-muted-foreground">
                          {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB • Ready for analysis
                        </p>
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={removeFile}
                      className="text-muted-foreground hover:text-destructive"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

          </div>
        </div>

        {/* Chat Messages */}
        {messages.length > 0 && (
          <Card className="flex-1 min-h-0">
            <CardContent className="p-0 h-full">
              <ScrollArea ref={scrollAreaRef} className="h-full">
                <div className="p-6 space-y-4">
                  {messages.map((message) => (
                    <PDFMessage key={message.id} message={message} />
                  ))}
                  
                  {isLoading && (
                    <div className="flex gap-3 p-3">
                      <div className="flex h-7 w-7 shrink-0 select-none items-center justify-center rounded-lg bg-primary text-primary-foreground">
                        <Loader2 className="h-3 w-3 animate-spin" />
                      </div>
                      <div className="flex flex-col space-y-1 text-sm">
                        <div className="px-3 py-2 rounded-xl rounded-bl-sm bg-secondary/50 text-foreground">
                          <div className="flex space-x-1">
                            <div className="w-1.5 h-1.5 bg-muted-foreground rounded-full typing-indicator"></div>
                            <div className="w-1.5 h-1.5 bg-muted-foreground rounded-full typing-indicator" style={{ animationDelay: '0.3s' }}></div>
                            <div className="w-1.5 h-1.5 bg-muted-foreground rounded-full typing-indicator" style={{ animationDelay: '0.6s' }}></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileInputChange}
          className="hidden"
        />
      </div>
    </div>
  );
};