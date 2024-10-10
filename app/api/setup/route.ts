import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";

import {
    createPineconeIndex,
    updatePinecone
} from '../../../utils'

import { indexName } from "@/config";
import { index } from "langchain/indexes";

export async function POST() {
    const loader = new DirectoryLoader('./documents', {
        ".txt" : (path) => new TextLoader(path),
        ".md" : (path) => new TextLoader(path),
        ".pdf" : (path) => new TextLoader(path),
    })
    const docs = await loader.load();
    const vectorDimensions = 1536

    const client = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY || '',
    })

    try {
        await createPineconeIndex(client, index, vectorDimensions)
        await updatePinecone(client, indexName, docs)
    } catch (error) {
        console.log('error: ', error);
    }

    return NextResponse.json({
        data: 'Successfully created index and loaded data into Pinecone... '
    })
}
