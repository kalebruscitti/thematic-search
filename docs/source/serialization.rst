Saving and Loading
==================

A ``thematic_search.TopicDatabase`` object supports saving and loading from disk. 
Suppose ``topicdb`` is a TopicDatabase. Then there are two options for 
saving to disk: ::

    topicdb.to_file("my-topicdatabase.tm.zip")

    topicdb.to_lance("my-topicdatabase")

The ``to_file()`` method saves the metadata as parquet files and the vector arrays 
as ``.npz`` files, then writes a metadata JSON file and zips everything for portability.

The ``to_lance()`` method saves everything to a `LanceDB`_ folder.

.. _LanceDB: https://docs.lancedb.com/

If you have a saved `tm.zip` file or LanceDB folder, you can load it using 
the appropriate class method: ::

    topicdb = TopicDatabase.from_file("my-topicdatabase.tm.zip")

    topicdb = TopicDatabase.from_lance("my-topicdatabase")

Note that the embedding model of a TopicDatabase is not saved.
You will have to reload your embedding model separately, and then
set `topicdb.embedding_model` manually. For example: ::

    import thematic_search as ts
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    topicdb = ts.TopicDatabase.from_file("docs/source/20ng-topicdb.tm.zip")
    topicdb.embedding_model = model