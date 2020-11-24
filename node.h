//
// Created by 田地 on 2020/9/22.
//

#ifndef SERVER_NODE_H
#define SERVER_NODE_H

#include <vector>
#include <string>

using std::vector;
using std::string;

namespace grp {
    class Graph;
}

namespace node {

    enum OpType {
        conv,
        bn,
        relu,
        gemm,
        flatten,
        shape,
        gather,
        mul,
        unsqueeze,
        concat,
        reshape,
        pad,
        avgpool,
        gavgpool,
        add,
        constant,
        unknown
    };

    class Node {
    private:
        vector<string> inputs;
        vector<string> outputs;
//        vector<Node&> succs;  // todo: this need to be done with shared pointers
        OpType type;
        grp::Graph& g;
        int id;
    public:
        Node(OpType t,
                int id,
                vector<string> inputs,
                vector<string> outputs,
                grp::Graph& g) : type(t), id(id), inputs(inputs), outputs(outputs), g(g) {};
        virtual ~Node() = default;
        inline const OpType Type() const { return this->type; }
        inline const vector<string> Inputs() const { return this->inputs; }
        inline const vector<string> Outputs() const { return this->outputs; }
        inline grp::Graph& GetGraph() { return this->g; }
        inline int ID() const { return this->id; }
        virtual bool Absorb(Node another);

    protected:
        inline void setOutputs(vector<string> newOutputs) {
            outputs = newOutputs;
        }
    };

    class ConvNode : public Node {
    private:
        int group;
        vector<int> dilations;
        vector<int> kernelShape;
        vector<int> pads; // l, r, u, d
        vector<int> strides;
        vector<int> weightDims;
        vector<int> biasDims;
        string srcInputName;
        string weightName;
        string biasName;
        bool relu;
        string sumName;
    public:
        virtual ~ConvNode() = default;
        ConvNode(
                OpType t,
                int id,
                vector<string> inputs,
                vector<string> outputs,
                grp::Graph& g,
                int group,
                vector<int> dilations,
                vector<int> kernelShape,
                vector<int> pads,
                vector<int> strides,
                vector<int> weightDims,
                vector<int> biasDims,
                string srcName,
                string weightName,
                string biasName);
        const vector<int>& KernelShape() const;
        const vector<int>& Pads() const;
        const vector<int>& Strides() const;
        const vector<int>& WeightDims() const;
        const vector<int>& BiasDims() const;
        const string SrcInputName() const;
        const string WeightName() const;
        const string BiasName() const;
        const string OutputName() const;
        virtual bool Absorb(std::shared_ptr<Node> another);
        void EnablePostRelu();
        inline void EnablePostSum(const string& outputName) {sumName = outputName;}
        inline const bool PostRelu() const {return relu;}
        inline const bool PostSum() const {return !sumName.empty();}
        inline const string PostSumOutputName() {return sumName;}
    };

    class BatchNormNode : public Node {
    private:
        float epsilon;
        float momentum; // factor used to compute running mean and standard
        vector<int> dim;
        string wightName;
        string biasName;
        string meanName;
        string varName;
        string srcInputName;
        bool relu;
    public:
        virtual ~BatchNormNode() = default;
        BatchNormNode(
                node::OpType t,
                int id,
                const vector<string> inputs,
                const vector<string> outputs,
                grp::Graph& g,
                float epsilon,
                float momentum,
                vector<int> dim,
                string wightName,
                string biasName,
                string meanName,
                string varName,
                string srcInputName);
        float Epsilon() const;
        float Momentum() const; // factor used to compute running mean and standard
        const vector<int>& Dim() const;
        string WightName() const;
        string BiasName() const;
        string MeanName() const;
        string VarName() const;
        string SrcInputName() const;
        string OutputName() const;
        virtual bool Absorb(std::shared_ptr<Node> another);
        inline void EnablePostRelu() {this->relu = true;}
        inline const bool PostRelu() const {return this->relu;}
    };

    // Note: we do not need relu node! The base node is enough!
    class ShapeNode : public Node {
    public:
        ShapeNode(OpType t,
                int id,
                vector<string> inputs,
                vector<string> outputs,
                grp::Graph& g)
                : Node(t, id, inputs, outputs, g) {};
        inline string InputName() const {return this->Inputs()[0];}
        inline string OutputName() const {return this->Outputs()[0];}
    };

    // Gather data on the given dimension
    // e.g, data is 6×5×4×3, indices is 2×1, axis = 0, then output is 2×1×5×4×3
    // e.g, data is 6×5×4×3, indices is 2×1, axis = 1, then output is 6×2×1×4×3
    // e.g, data is 6×5×4×3, indices is 2×1, axis = 2, then output is 6×5×2×1×3
    // e.g, data is 6×5×4×3, indices is 2×1, axis = 0, then output is 6×5×4×2×1
    // Why this is usefule?
    // App1: when we flat matrix, we need to take each of the dimsions,
    // and multiply them all, gather can be used to take out a certain dim
    // App2: we may need to construct new features by combining original features in different way
    // todo: negative index
    class GatherNode : public Node {
    private:
        int axis;
    public:
        GatherNode(OpType t,
                  int id,
                  vector<string> inputs,
                  vector<string> outputs,
                   grp::Graph& g,
                  int axis)
                : Node(t, id, inputs, outputs, g), axis(axis) {};
        inline int Axis() const {return this->axis;}
        inline string DataName() const {return this->Inputs()[0];}
        inline string IndicesName() const {return this->Inputs()[1];}
        inline string OutputName() const {return this->Outputs()[0];}
    };

    // element-wise multiplication
    //
    class MulNode : public Node {
    public:
        MulNode(OpType t,
                   int id,
                   vector<string> inputs,
                   vector<string> outputs,
                   grp::Graph& g)
                : Node(t, id, inputs, outputs, g) {};
        // For onnx
        // A, B could come from initial weights or output from other nodes
        inline string InputAName() const {return this->Inputs()[0];}
        inline string InputBName() const {return this->Inputs()[1];}
        inline string OutputName() const {return this->Outputs()[0];}
    };

    // squeeze all single dimensional entries
    class UnsqueezeNode : public Node {
    private:
        vector<int> axes; // not used at this time
    public:
        UnsqueezeNode(OpType t,
                int id,
                vector<string> inputs,
                vector<string> outputs,
                grp::Graph& g)
                : Node(t, id, inputs, outputs, g) {};
        inline string InputName() const {return this->Inputs()[0];}
        inline string OutputName() const {return this->Outputs()[0];}
    };

    class ConcatNode : public Node {
    private:
        int axis;
    public:
        ConcatNode(OpType t,
              int id,
              vector<string> inputs,
              vector<string> outputs,
              grp::Graph& g,
              int axis)
                : Node(t, id, inputs, outputs, g), axis(axis) {};
        inline vector<string> InputsName() const {return this->Inputs();}
        inline string OutputName() const {return this->Outputs()[0];}
        inline int Axis() const {return this->axis;}
    };

    class ReshapeNode : public Node {
    public:
        ReshapeNode(OpType t,
                   int id,
                   vector<string> inputs,
                   vector<string> outputs,
                   grp::Graph& g)
                : Node(t, id, inputs, outputs, g) {};
        inline string DataName() const {return this->Inputs()[0];}
        inline string ShapeName() const {return this->Inputs()[1];}
        inline string OutputName() const {return this->Outputs()[0];}
    };

    class FlattenNode : public Node {
    private:
        int axis;
    public:
        FlattenNode(OpType t,
                   int id,
                   vector<string> inputs,
                   vector<string> outputs,
                   grp::Graph& g,
                   int axis)
                : Node(t, id, inputs, outputs, g), axis(axis) {};
        inline string InputsName() const {return this->Inputs()[0];}
        inline string OutputName() const {return this->Outputs()[0];}
        inline int Axis() const {return this->axis;}
    };

    class GemmNode : public Node {
    private:
        float alpha;
        float beta;
        bool transA;
        bool transB;
        bool bias;
    public:
        GemmNode(OpType t,
                int id,
                vector<string> inputs,
                vector<string> outputs,
                grp::Graph& g,
                float alpha,
                float beta,
                int transA,
                int transB,
                bool bias)
                : Node(t, id, inputs, outputs, g), alpha(alpha), beta(beta),transA(transA), transB(transB), bias(bias) {};
        inline string InputName() const {return this->Inputs()[0];}
        inline string WeightName() const {return this->Inputs()[1];}
        inline string BiasName() const {return this->Inputs()[2];}
        inline string OutputName() const {return this->Outputs()[0];}
        inline float Alpha() const {return this->alpha;}
        inline float Beta() const {return this->beta;}
        inline bool TransA() const {return this->transA;}
        inline bool TransB() const {return this->transB;}
        inline bool Bias() const {return this->bias;}
    };

    class PadNode : public Node {
    private:
        string mode;
        vector<int> pads;
        float value;
    public:
        PadNode(OpType t,
                int id,
                vector<string> inputs,
                vector<string> outputs,
                grp::Graph& g,
                string& mode,
                vector<int>& pads,
                float value)
                : Node(t, id, inputs, outputs, g), mode(mode), pads(pads), value(value) {};
        inline string InputName() const {return this->Inputs()[0];}
        inline string OutputName() const {return this->Outputs()[0];}
        inline string Mode() const {return this->mode;}
        inline const vector<int>& Pads() const {return this->pads;}
        inline float Value() const {return this->value;}
    };

    class AvgPoolingNode : public Node {
    private:
        vector<int> kernelShape;
        vector<int> pads;
        vector<int> strides;
    public:
        AvgPoolingNode(OpType t,
                int id,
                vector<string> inputs,
                vector<string> outputs,
                grp::Graph& g,
                vector<int>& kernelShape,
                vector<int>& pads,
                vector<int>& strides)
                : Node(t, id, inputs, outputs, g), kernelShape(kernelShape), pads(pads), strides(strides) {};
        inline string InputName() const {return this->Inputs()[0];}
        inline string OutputName() const {return this->Outputs()[0];}
        inline const vector<int>& KernelShape() const {return this->kernelShape;}
        inline const vector<int>& Pads() const {return this->pads;}
        inline const vector<int>& Strides() const {return this->strides;}
    };

    class GlobalAvgPoolingNode : public Node {
    public:
        GlobalAvgPoolingNode(OpType t,
                       int id,
                       vector<string> inputs,
                       vector<string> outputs,
                       grp::Graph& g)
                : Node(t, id, inputs, outputs, g) {};
        inline string InputName() const {return this->Inputs()[0];}
        inline string OutputName() const {return this->Outputs()[0];}
    };

    class AddNode : public Node {
    public:
        AddNode(OpType t,
                int id,
                vector<string> inputs,
                vector<string> outputs,
                grp::Graph& g)
                : Node(t, id, inputs, outputs, g) {};
        inline string InputA() const {return this->Inputs()[0];}
        inline string InputB() const {return this->Inputs()[1];}
        inline string OutputName() const {return this->Outputs()[0];}
    };

    class ConstantNode : public Node {
    private:
    public:
        ConstantNode(OpType t,
                int id,
                vector<string> inputs,
                vector<string> outputs,
                grp::Graph& g)
                : Node(t, id, inputs, outputs, g) {};
        inline string OutputName() const {return this->Outputs()[0];}
    };
}

#endif //SERVER_NODE_H
