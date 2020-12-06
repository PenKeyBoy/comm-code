#!/usr/bin/env/py35
# coding=utf-8
from time import time
import hashlib
import json
from uuid import uuid4
from flask import Flask,jsonify,request
import requests
from urllib.parse import urlparse

class BlockChain(object):
    #construct method info about chain and current transaction
    def __init__(self):
        self.chain = []
        self.current_transactions = []
        #generate genisis block(the origin block)
        self.new_block(1,previous_hash=100)
        #specify the valid proof is or not
        #add nodes info for different nodes interative
        self.nodes = set()

    def register_node(self,address):
        """
        add address info into nodes by parse the address
        :param address: <string> parse node info from which format like eg: http://192.167.185.1/...
        :return: None
        """
        pu = urlparse(address)
        self.nodes.add(pu.netloc)

    def resolve_conflicts(self):
        """
        choose the max length chain according to spread over all nodes
        :return: boolean find the max length chain:True else:False
        """
        new_chain = None
        max_len = len(self.chain)
        print("max_len=",max_len)
        neighbor_nodes = self.nodes
        #grab and verify the chain from all of the neighbor_nodes
        for node in neighbor_nodes:
            #get ask for node from neighbor_nodes
            try:
                response = requests.get('http://%s/chain' % node)
            except:
                continue
            if response.status_code == 200:
                print("get chain info")
                length = response.json()['length']
                print("length chain from neighbor_node={}".format(length))
                chain = response.json()['chain']
                if length > max_len and self.valid_chain(chain):
                    print("new chain generate")
                    max_len = length
                    new_chain = chain
        #if exists the new_chain,update the chain info
        if new_chain:
            self.chain = new_chain
            return True
        return False


    def valid_chain(self,chain):
        """
        valid the chain if significant or not
        :param chain: chain to be valid <blockChain object attribute>
        :return: True if go accross the valid result or False
        """
        current_index = 1
        last_block = chain[0]

        while current_index < len(chain):
            print("the last_block of the chain:{}".format(chain[-1]))
            print("current block info:{}".format(chain[current_index]))
            print("\n------------------\n")
            block = chain[current_index]
            previous_hash = block['previous_hash']
            proof = block['proof']
            last_proof = last_block['proof']
            if previous_hash != self.hash(last_block):
                print("hash value not equal in current_index=",current_index)
                return False
            if not self.valid_proof(proof,last_proof):
                print("proof can't reach effective in current_index=",current_index)
                return False


            last_block = chain[current_index]
            current_index += 1
        return True

    def new_transaction(self,sender,recipator,amout):
        """
        describe the new transaction info
        :param sender: <string> sender info
        :param recipator: <string> recipator info
        :param amout: <int>
        :return: <int> the index of last block index + 1
        """
        self.current_transactions.append({
            "sender":sender,
            "recipator":recipator,
            "amout":amout
        })
        return self.last_block['index'] + 1
    
    def new_block(self,proof,previous_hash=None):
        """
        construction of block format
        :param proof: <int> proof of work
        :param previous_hash: <string>last block proof hash value
        :return: block<dict/json>
        """
        block = {
            'index':len(self.chain) + 1,
            'timestamp':time(),
            'transactions':self.current_transactions,
            'proof':proof,
            'previous_hash':previous_hash or self.hash(self.last_block)
        }
        #reset the current_transactions
        self.current_transactions = []
        self.chain.append(block)

        return block
    @property
    def last_block(self):
        return self.chain[-1]

    @staticmethod
    def hash(block):
        """
        map the block into hash value
        :param block: <dict/json>
        :return: <string> hash value
        """
        block_string = json.dumps(block,sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def proof_of_work(self,last_proof):
        """
        according to valid the proof to specify the proof value
        :param last_proof: <int> last block proof result
        :return: <int>
        """
        proof = 0
        while not self.valid_proof(proof,last_proof):
            proof += 1
        return proof
    @staticmethod
    def valid_proof(proof,last_proof):
        """
        specify the word is proof or not
        :param proof: <int> current proof
        :param last_proof: <int> last proof
        :return: <boolean>
        """
        string_proof = str(proof * last_proof).encode()
        hash_value = hashlib.sha256(string_proof).hexdigest()
        return hash_value[:4] == '0'*4

app = Flask(__name__)

block = BlockChain()
node_identify = str(uuid4()).replace('-','')

@app.route('/transactions/new',methods=['POST'])
def new_transactions():
    #create new transactions for my block
    transaction = request.get_json()
    check_tran = ['sender','recipator','amout']
    if not all(key in transaction for key in check_tran):
        return '201 error: missing value for transaction'
    index = block.new_transaction(transaction['sender'],transaction['recipator'],transaction['amout'])
    response = {'message':'new transaction will be add to block {}'.format(index)}
    return jsonify(response),202
    #return "we will add new transaction to block"
@app.route('/mine',methods=['GET'])
def mine():
    #use proof_of_work to valid
    last_block = block.last_block
    last_proof = last_block['proof']
    proof = block.proof_of_work(last_proof)
    transcation_info = {
        'sender':'0',
        'recipator':node_identify,
        'amout':1
    }
    block.new_transaction(transcation_info['sender'],transcation_info['recipator'],transcation_info['amout'])
    previous_hash = block.hash(last_block)
    block_info = block.new_block(proof,previous_hash=previous_hash)
    response = {
        'message':'new block forged',
        'index':block_info['index'],
        'transactions':block_info['transactions'],
        'proof':block_info['proof'],
        'previous_hash':block_info['previous_hash']
    }
    return jsonify(response),201

@app.route('/chain',methods=['GET'])
def full_chain():
    # 返回完整的链信息,以及区块的长度
    response = {
        'chain':block.chain,
        'length':len(block.chain)
    }
    return jsonify(response),200

#register nodes info
@app.route('/nodes/register',methods=['POST'])
def register_node():
    nodes_info = request.get_json()
    nodes = nodes_info.get("nodes")
    if not nodes:
        return "Error:no nodes add into network",201
    for node in nodes:
        node = block.register_node(node)
        if node:
            block.nodes.add(node)
    response = {
        "message": "we have add new nodes into",
        "nodes": list(block.nodes)
    }
    return jsonify(response),200

@app.route('/conflict/resolve',methods=['GET'])
def resolve_conflict():
    replaced = block.resolve_conflicts()
    if replaced:
        response = {
            "message": "our chain is replaced",
            "new_chain": block.chain
        }
    else:
        response = {
            "message": "our chain is authoritative",
            "chain": block.chain
        }
    return jsonify(response),300

if __name__ == '__main__':
    app.run('127.0.0.1',5000)