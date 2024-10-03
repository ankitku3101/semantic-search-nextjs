import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAI } from '@langchain//openai'
import { loadQAStuffChain } from "langchain/chains";
import { Document } from "langchain/document";
import { timeout } from './config'
import { log } from "console";
import { metadata } from "./app/layout";

export const createPinecodeIndex = async (
    client,
    indexName,
    vectorDimension
) => {
    console.log(`Checking "${indexName}"...`);
    const exisitingIndexes = await client.listIndexes();
    if(!exisitingIndexes.includes(indexName)) {
        console.log(`Creating "${indexName}"...`);
        
        await client.createIndex({
            createRequest: {
                name: indexName,
                dimension: vectorDimension,
                metric: 'cosine'        
            }
        })
        console.log(`Creating index... please wait for it to finish initializing. `);
        await new Promise((resolve) => setTimeout(resolve, timeout));
    } else {
        console.log(`"${indexName}" already exists.`);       
    }
}

export const updatePinecone = async (client, indexName, docs) => {
    const index = client.index(indexName)
    console.log(`Pinecone index retrieved: ${indexName}`);
    
    for(const doc of docs) {
        console.log(`Processing document ${doc.metadata.source}`);
        const txtPath = doc.metadata.source
        const text = doc.pageContent
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
        })
        console.log('Splitting text into chunks...');
        const chunks = await textSplitter.createDocuments([text]);
        console.log(`Text split into ${chunks.length} chunks`);
        console.log(`Calling OpenAI's Embedding endpoint documents with ${chunks.length} text chunks...`);
        const embeddingsArray = await new OpenAIEmbeddings().embedDocuments(
            chunks.map((chunk) => chunk.pageContent.replace(/\n/g, " "))
        );
        console.log(`Creating ${chunks.length} vectors array with id, values, and metadata...`);
        
        const batchSize = 100
        let batch:any = [];
        for (let idx = 0; idx < chunks.length; idx++) {
            const chunk = chunks[idx];
            const vector = {
                id: `${txtPath}_${idx}`,
                values: embeddingsArray[idx],
                metadata: {
                    ...chunk.metadata,
                    loc: JSON.stringify(chunk.metadata.loc),
                    pageContent: chunk.pageContent,
                    txtPath: txtPath    
                }
            }
        }
    }
}