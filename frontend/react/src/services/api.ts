import axios from "axios";

const API_URL = "http://localhost:8000"; // FastAPI backend

export const chatPdf = async (message: string) => {
  const res = await axios.post(`${API_URL}/chat_pdf`, { message });
  return res.data.reply;
};
