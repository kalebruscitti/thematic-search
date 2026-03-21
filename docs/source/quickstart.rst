Quickstart
==========

Installation
------------

This is an alpha version, so one still must install from source::

    git clone git@github.com:kalebruscitti/thematic-search.git
    pip install thematic-search

Basic Usage
-----------

To use Thematic Search, you need a hierarchical topic model of your dataset. The
minimal ingredients are:

- ``embedding_vectors``: an ``(n_docs, d)`` float array of document embeddings
- ``cluster_tree``: a dictionary ``{node: [children]}`` representing your topic
  hierarchy, where nodes can be any hashable labels (strings, ints, etc.)
- ``cluster_layers``: a list of ``(n_docs, n_clusters)`` float arrays in ``[0,1]``,
  one per layer, where ``cluster_layers[l][j, i]`` is the inclusion strength of
  document ``j`` in the ``i``-th cluster at layer ``l``

Optionally you can also provide:

- ``topic_metadata``: a ``DataFrame`` with a row for each node in ``cluster_tree``,
  indexed by the same node labels
- ``document_metadata``: a ``DataFrame`` with a row for each document
- ``reduced_vectors``: an ``(n_docs, 2)`` array of low-dimensional vectors for
  visualisation

Converting your cluster tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your ``cluster_tree`` can use your own node labels. The
``convert_tree`` utility converts it into the ``(layer, index)`` tuple format
required by ``SoftClusterTree``, and returns a ``cluster_labels`` mapping that
lets ``TopicDatabase`` automatically align your ``topic_metadata``::

    from thematic_search.utilities import convert_tree

    # Example: a simple tree with string node labels
    my_tree = {
        "root":   ["science", "sports"],
        "science": ["physics", "biology"],
        "sports":  ["football", "tennis"],
        "physics": [], "biology": [], "football": [], "tennis": [],
    }

    cluster_tree, cluster_labels = convert_tree(my_tree)

If your nodes are not naturally arranged into layers, ``convert_tree`` will
assign layers automatically: leaves get layer 0, and each internal node gets
one layer above its deepest child. You can override this by passing a custom
``layers`` dictionary::

    cluster_tree, cluster_labels = convert_tree(my_tree, layers={"root": 3, ...})

Initializing a TopicDatabase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``cluster_labels`` to ``TopicDatabase`` alongside your metadata. If you
provide a ``topic_metadata`` DataFrame indexed by your original node labels,
it will be re-indexed automatically::

    from thematic_search import TopicDatabase, SoftClusterTree

    topicdb = TopicDatabase(
        SoftClusterTree(cluster_layers, cluster_tree),
        embedding_vectors=embedding_vectors,
        reduced_vectors=reduced_vectors,          # optional
        document_df=document_metadata,            # optional
        topic_df=topic_metadata,                  # indexed by your node labels
        cluster_labels=cluster_labels,            # from convert_tree
    )

Querying
--------

The query interface is accessed via ``topicdb.q``. Queries are chainable and
follow the arrows in the database schema: you can start from a text string, a
set of document indices, or a known topic, and navigate between documents and
topics using the methods below.

Semantic search
~~~~~~~~~~~~~~~

Find the documents nearest to a query string in embedding space::

    topicdb.q.search("Advancements in space technology").documents()

This requires an ``embedding_model`` property to be provided to the TopicDatabase.

Thematic search
~~~~~~~~~~~~~~~

Find the most specific topic that best covers the nearest neighbours of a
query string::

    topicdb.q.search("Advancements in space technology").theme().info()

Find all documents inside a given topic with at least 75% inclusion strength::

    topicdb.q.topic_name("science").inside(min_strength=0.75).documents()

Chaining queries
~~~~~~~~~~~~~~~~

Queries can be chained arbitrarily. For example, to find the theme of the
documents inside the parent of a known topic::

    topicdb.q.topic_name("physics").parents().inside().theme().info()